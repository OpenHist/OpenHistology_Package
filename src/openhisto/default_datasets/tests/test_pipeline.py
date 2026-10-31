import contextlib
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openhisto.default_datasets import (
    AcquisitionPlan,
    Asset,
    MetadataRecord,
    build_manifest,
    import_local,
)
from openhisto.default_datasets.__main__ import main


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="cpath-dataset-test-")
        self.root = Path(self.temporary.name)
        (self.root / "existing").mkdir()
        (self.root / "existing" / "one.svs").write_bytes(b"synthetic-slide-one")
        (self.root / "existing" / "two.svs").write_bytes(b"synthetic-slide-two")

    def tearDown(self):
        self.temporary.cleanup()

    def plan(self):
        return AcquisitionPlan(
            "TCGA",
            {"projects": ["TCGA-SYNTHETIC"]},
            assets=[
                Asset(
                    "file-one",
                    "TCGA",
                    "slide",
                    "one.svs",
                    "slides/file-one.svs",
                    slide_id="slide-one",
                    subject_ids=["case-one"],
                    clinical_refs=["case:case-one"],
                    metadata={"source_mag": 20},
                ),
                Asset(
                    "file-two",
                    "TCGA",
                    "slide",
                    "two.svs",
                    "slides/file-two.svs",
                    slide_id="slide-two",
                    subject_ids=["case-missing"],
                ),
            ],
            records=[
                MetadataRecord(
                    "case:case-one",
                    "clinical",
                    ["case-one"],
                    {
                        "case_id": "case-one",
                        "diagnoses": [{"diagnosis_id": "d1"}, {"diagnosis_id": "d2"}],
                        "follow_ups": [{"days_to_follow_up": 12}, {"days_to_follow_up": 24}],
                    },
                )
            ],
            provenance={"release": "synthetic-v1"},
        )

    def test_local_registration_preserves_nested_metadata_and_missingness(self):
        report = import_local("tcga", self.root, self.plan())
        self.assertEqual((report.valid, report.missing, report.failed), (2, 0, 0))
        slides = [
            json.loads(line) for line in (self.root / "manifest.jsonl").read_text().splitlines()
        ]
        self.assertEqual(slides[0]["local_path"], "existing/one.svs")
        self.assertEqual(slides[0]["group_ids"], ["TCGA:case-one"])
        self.assertEqual(slides[0]["clinical_status"], "available")
        self.assertEqual(slides[1]["clinical_status"], "missing")
        self.assertEqual(slides[1]["label_status"], "unavailable")
        self.assertIsNone(slides[1]["source_mag"])
        clinical = json.loads((self.root / "clinical.jsonl").read_text())
        self.assertEqual(len(clinical["data"]["diagnoses"]), 2)
        self.assertEqual(len(clinical["data"]["follow_ups"]), 2)
        self.assertNotIn("survival", clinical["data"])
        self.assertFalse((self.root / "slides").exists())
        self.assertEqual(len(list(self.root.rglob("*.svs"))), 2)

    def test_conflicting_source_records_are_retained(self):
        plan = self.plan()
        plan.records.append(
            MetadataRecord(
                "case:case-one",
                "clinical",
                ["case-one"],
                {"case_id": "case-one", "different_record": True},
            )
        )
        import_local("TCGA", self.root, plan)
        slides = [
            json.loads(line) for line in (self.root / "manifest.jsonl").read_text().splitlines()
        ]
        self.assertEqual(slides[0]["clinical_status"], "conflicting")
        self.assertEqual(len(slides[0]["clinical_refs"]), 2)
        self.assertEqual(len((self.root / "clinical.jsonl").read_text().splitlines()), 2)

    def test_identical_clinical_records_keep_both_sources_without_false_conflict(self):
        plan = self.plan()
        original = plan.records[0]
        original.source = "synthetic-source-a.csv"
        plan.records.append(
            MetadataRecord(
                original.record_id,
                original.kind,
                list(original.subject_ids),
                dict(original.data),
                "synthetic-source-b.csv",
            )
        )
        import_local("TCGA", self.root, plan)
        slide = json.loads((self.root / "manifest.jsonl").read_text().splitlines()[0])
        self.assertEqual(slide["clinical_status"], "available")
        self.assertEqual(len(slide["clinical_refs"]), 1)
        clinical = json.loads((self.root / "clinical.jsonl").read_text())
        self.assertEqual(clinical["sources"], ["synthetic-source-a.csv", "synthetic-source-b.csv"])

    def test_changed_saved_plan_cannot_inherit_stale_completion_state(self):
        import_local("TCGA", self.root, self.plan())
        plan = AcquisitionPlan.load(self.root / "acquisition_plan.json")
        plan.assets[0].relative_path = "new-location/one.svs"
        plan.save(self.root / "acquisition_plan.json")
        manifest = build_manifest(self.root)
        slide = json.loads(manifest.read_text().splitlines()[0])
        self.assertEqual(slide["local_path"], "new-location/one.svs")
        self.assertEqual(slide["status"], "pending")
        self.assertIsNone(slide["local_sha256"])

    def test_missing_local_files_and_rebuild(self):
        (self.root / "existing" / "two.svs").unlink()
        report = import_local("TCGA", self.root, self.plan())
        self.assertEqual(report.missing, 1)
        path = build_manifest(self.root)
        slides = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(slides[1]["status"], "missing")

    def test_ambiguous_local_names_fail_without_state(self):
        (self.root / "other").mkdir()
        (self.root / "other" / "one.svs").write_bytes(b"other-synthetic-slide")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            import_local("TCGA", self.root, self.plan())
        self.assertFalse((self.root / "acquisition_plan.json").exists())

    def test_explicit_selection_and_plan_roundtrip(self):
        from openhisto.default_datasets import discover

        with self.assertRaises(ValueError):
            discover("TCGA", {})
        original = self.plan()
        path = original.save(self.root / "selection.json")
        self.assertEqual(AcquisitionPlan.load(path).to_dict(), original.to_dict())

    def test_manifest_can_feed_patch_extraction_without_clinical_ids_in_archive(self):
        import numpy as np
        from PIL import Image
        from openhisto.patch_extraction import ExtractionConfig, PatchReader, extract_grid

        class SyntheticSlide:
            dimensions = (32, 16)
            level_downsamples = (1.0,)
            properties = {"openslide.objective-power": "20"}

            def read_region(self, location, level, size):
                return Image.fromarray(
                    np.full((size[1], size[0], 4), (160, 90, 140, 255), dtype=np.uint8)
                )

            def close(self):
                pass

        import_local("TCGA", self.root, self.plan())
        slide = json.loads((self.root / "manifest.jsonl").read_text().splitlines()[0])
        with patch(
            "openhisto.patch_extraction.extraction._open_slide", return_value=SyntheticSlide()
        ):
            extract_grid(
                self.root / slide["local_path"],
                self.root / "patches.h5",
                ExtractionConfig(patch_size=16, source_mag=slide["source_mag"]),
            )
        with PatchReader(self.root / "patches.h5") as reader:
            self.assertEqual(len(reader), 2)
            self.assertNotIn("subject_ids", reader.metadata)
            self.assertNotIn("clinical_refs", reader.metadata)

    def test_kpmp_downloaded_clinical_table_is_linked_without_unbinning(self):
        table = self.root / "published_clinical.csv"
        table.write_text(
            "Participant ID,Age,eGFR,Unknown source column\nsynthetic-subject,30-39,60-89,retained\n",
            encoding="utf-8",
        )
        repository = self.root / "atlas.csv"
        with repository.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "Access",
                    "Participant ID",
                    "File Name",
                    "Internal Package ID",
                    "Experimental Strategy",
                    "Data Category",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "Access": "open",
                        "Participant ID": "synthetic-subject",
                        "File Name": "one.svs",
                        "Internal Package ID": "synthetic-package",
                        "Experimental Strategy": "Light Microscopic Whole Slide Images",
                        "Data Category": "Pathology",
                    },
                    {
                        "Access": "open",
                        "Participant ID": "",
                        "File Name": table.name,
                        "Internal Package ID": "synthetic-clinical-package",
                        "Experimental Strategy": "Clinical Study Data",
                        "Data Category": "Clinical",
                    },
                ]
            )
        report = import_local("KPMP", self.root, {"repository_csv": str(repository)})
        self.assertEqual(report.valid, 2)
        slide = json.loads((self.root / "manifest.jsonl").read_text())
        self.assertEqual(slide["clinical_status"], "available")
        self.assertEqual(slide["subject_ids"], ["synthetic-subject"])
        record = json.loads((self.root / "clinical.jsonl").read_text())
        self.assertEqual(record["data"]["Age"], "30-39")
        self.assertEqual(record["data"]["Unknown source column"], "retained")

    def test_bracs_common_manifest_preserves_split_and_unavailable_clinical(self):
        table = self.root / "bracs.csv"
        table.write_text(
            "WSI,WSI label,Patient ID,Set\none,ADH,synthetic-subject,validation\n", encoding="utf-8"
        )
        report = import_local("BRACS", self.root, str(table))
        self.assertEqual(report.valid, 2)
        slides = [
            json.loads(line) for line in (self.root / "manifest.jsonl").read_text().splitlines()
        ]
        labeled = next(slide for slide in slides if slide["slide_id"] == "one")
        self.assertEqual(labeled["labels"], {"wsi_label": "ADH"})
        self.assertEqual(labeled["split"], "validation")
        self.assertEqual(labeled["clinical_status"], "unavailable")
        self.assertEqual((self.root / "clinical.jsonl").read_text(), "")

    def test_camelyon_common_manifest_retains_labels_without_inventing_subjects(self):
        (self.root / "existing" / "synthetic.tif").write_bytes(b"synthetic-native-tiff")
        plan = AcquisitionPlan(
            "CAMELYON16",
            {"slide_ids": ["synthetic"]},
            assets=[
                Asset(
                    "synthetic",
                    "CAMELYON16",
                    "slide",
                    "synthetic.tif",
                    "slides/synthetic.tif",
                    slide_id="synthetic",
                    labels={"source_class": "normal"},
                    label_status="known",
                    split="train",
                ),
            ],
        )
        report = import_local("CAMELYON16", self.root, plan)
        self.assertEqual(report.valid, 1)
        slide = json.loads((self.root / "manifest.jsonl").read_text())
        self.assertEqual(slide["format"], "tif")
        self.assertEqual(slide["labels"], {"source_class": "normal"})
        self.assertEqual(slide["subject_ids"], [])
        self.assertEqual(slide["group_ids"], [])
        self.assertEqual(slide["clinical_status"], "unavailable")

    def test_manifest_outputs_cannot_escape_through_symlink_parent(self):
        import_local("TCGA", self.root, self.plan())
        shutil_source = self.root / "source_metadata"
        for path in shutil_source.iterdir():
            path.unlink()
        shutil_source.rmdir()
        with TemporaryDirectory() as external:
            shutil_source.symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "within"):
                build_manifest(self.root)
            self.assertEqual(list(Path(external).iterdir()), [])

    def test_credentials_and_partial_plan_saves_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "credentials"):
            AcquisitionPlan(
                "KPMP", {"repository_csv": "synthetic.csv", "token": "synthetic-secret"}
            )
        plan = self.plan()
        plan.selection["password"] = "synthetic-secret"
        with self.assertRaises(ValueError):
            plan.save(self.root / "unsafe-plan.json")
        self.assertFalse((self.root / "unsafe-plan.json").exists())


class CliTests(unittest.TestCase):
    def test_local_import_cli_writes_real_manifests(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "synthetic.svs").write_bytes(b"synthetic-native-slide")
            table = root / "labels.csv"
            table.write_text(
                "WSI,WSI label,Patient ID,Set\nsynthetic,ADH,synthetic-group,validation\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    ["import-local", "bracs", "--root", str(root), "--metadata", str(table)]
                )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["valid"], 1)
            manifest = json.loads((root / "manifest.jsonl").read_text())
            self.assertEqual(manifest["split"], "validation")
            self.assertEqual(manifest["local_path"], "synthetic.svs")

    def test_discover_summary_does_not_dump_case_records(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            selection.write_text('{"projects":["TCGA-SYNTHETIC"]}')
            plan = AcquisitionPlan(
                "TCGA",
                {"projects": ["TCGA-SYNTHETIC"]},
                records=[
                    MetadataRecord(
                        "synthetic-case",
                        "clinical",
                        [],
                        {"private_fixture_field": "synthetic-value"},
                    )
                ],
            )
            stdout = io.StringIO()
            with (
                patch("openhisto.default_datasets.__main__.discover", return_value=plan),
                contextlib.redirect_stdout(stdout),
            ):
                self.assertEqual(
                    main(
                        [
                            "discover",
                            "tcga",
                            "--selection",
                            str(selection),
                            "--plan",
                            str(root / "plan.json"),
                        ]
                    ),
                    0,
                )
            self.assertNotIn("private_fixture_field", stdout.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())["slide_count"], 0)


if __name__ == "__main__":
    unittest.main()
