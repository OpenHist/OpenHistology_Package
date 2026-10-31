"""Provider contracts exercised only with synthetic HTTP fixtures."""

from copy import deepcopy
import html
import json
import unittest

from openhisto.default_datasets.CAMELYON16.provider import (
    BASE_URL,
    PREFIX,
    discover as discover_camelyon,
)
from openhisto.default_datasets.TCGA.provider import API_BASE, discover as discover_tcga


class Response:
    def __init__(self, body=None, *, text=None, status=200, headers=None):
        self.body = body
        self.content = (text if text is not None else json.dumps(body)).encode("utf-8")
        self.status_code = status
        self.headers = headers or {}

    def json(self):
        return deepcopy(self.body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"synthetic HTTP {self.status_code}")


class Session:
    def __init__(self, *, posts=(), gets=None):
        self.posts = list(posts)
        self.gets = gets or {}
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, deepcopy(kwargs)))
        if not self.posts:
            raise AssertionError("Unexpected metadata POST")
        expected_endpoint, response = self.posts.pop(0)
        if not url.endswith("/" + expected_endpoint):
            raise AssertionError(f"Expected {expected_endpoint}, received {url}")
        return response

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, deepcopy(kwargs)))
        params = kwargs.get("params", {})
        key = (url, params.get("prefix"), params.get("continuation-token")) if params else url
        if key not in self.gets:
            raise AssertionError(f"Unexpected metadata GET: {key}")
        return self.gets[key]


def gdc(hits, total=None, **extra):
    return Response(
        {
            "data": {"hits": hits, "pagination": {"total": len(hits) if total is None else total}},
            **extra,
        }
    )


def slide(file_id, cases, **overrides):
    result = {
        "file_id": file_id,
        "file_name": file_id + ".svs",
        "file_size": 1234,
        "md5sum": "a" * 32,
        "access": "open",
        "data_category": "Biospecimen",
        "data_type": "Slide Image",
        "data_format": "SVS",
        "experimental_strategy": "Diagnostic Slide",
        "cases": [
            {
                "case_id": case_id,
                "submitter_id": "synthetic-case-name",
                "samples": [{"sample_id": "sample-" + case_id}],
            }
            for case_id in cases
        ],
        "associated_entities": [
            {"entity_id": "slide-" + file_id, "entity_type": "slide", "case_id": case_id}
            for case_id in cases
        ],
    }
    result.update(overrides)
    return result


def listing(objects, *, next_token=None):
    entries = "".join(
        f"<Contents><Key>{html.escape(key)}</Key><Size>{size}</Size><ETag>{html.escape(etag)}</ETag></Contents>"
        for key, size, etag in objects
    )
    continuation = (
        f"<NextContinuationToken>{html.escape(next_token)}</NextContinuationToken>"
        if next_token
        else ""
    )
    return Response(
        text=(
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            + entries
            + f"<IsTruncated>{str(bool(next_token)).lower()}</IsTruncated>"
            + continuation
            + "</ListBucketResult>"
        )
    )


def cam_session(
    *,
    reference="image,type,split\nnormal_001.tif,normal,train\ntest_001.tif,tumor,test\n",
    checksum=None,
    annotations=False,
):
    gets = {
        BASE_URL + "/" + PREFIX + "README.md": Response(text="Synthetic dataset README"),
        BASE_URL + "/" + PREFIX + "license.txt": Response(text="Synthetic license text"),
        BASE_URL + "/" + PREFIX + "checksums.md5": Response(
            text=checksum if checksum is not None else "b" * 32 + "  images/normal_001.tif\n"
        ),
        BASE_URL + "/" + PREFIX + "evaluation/reference.csv": Response(text=reference),
        (BASE_URL, PREFIX + "images/", None): listing(
            [(PREFIX + "images/normal_001.tif", 100, '"' + "e" * 32 + '"')],
            next_token="synthetic-page-2",
        ),
        (BASE_URL, PREFIX + "images/", "synthetic-page-2"): listing(
            [
                (PREFIX + "images/test_001.tif", 200, '"' + "f" * 32 + '-2"'),
                (PREFIX + "images/tumor_001.tif", 300, '"' + "d" * 32 + '"'),
            ]
        ),
    }
    if annotations:
        gets[(BASE_URL, PREFIX + "annotations/", None)] = listing(
            [
                (PREFIX + "annotations/normal_001.xml", 15, '"annotation-etag"'),
                (PREFIX + "annotations/test_001.xml", 16, '"annotation-etag"'),
            ]
        )
        gets[(BASE_URL, PREFIX + "masks/", None)] = listing(
            [
                (PREFIX + "masks/test_001_mask.tif", 25, '"mask-etag"'),
                (PREFIX + "masks/tumor_001_mask.tif", 26, '"mask-etag"'),
            ]
        )
    return Session(gets=gets)


class TCGATests(unittest.TestCase):
    def test_paginated_file_discovery_and_all_case_links(self):
        case_a = {
            "case_id": "case-a",
            "demographic": {"vital_status": "Alive"},
            "diagnoses": [{"diagnosis_id": "diagnosis-a"}],
            "follow_ups": [{"days_to_follow_up": 42}],
        }
        case_b = {
            "case_id": "case-b",
            "demographic": {"vital_status": "Not Reported"},
            "exposures": [],
        }
        first = slide("file-a", ["case-a", "case-b"])
        first["associated_entities"][1]["entity_id"] = "second-slide-entity"
        session = Session(
            posts=[
                ("files", gdc([first], total=2, data_release="synthetic-release")),
                ("files", gdc([slide("file-b", ["case-a"])], total=2)),
                ("cases", gdc([case_a], total=2)),
                ("cases", gdc([case_b], total=2)),
            ]
        )
        selection = {"projects": ["TCGA-SYNTHETIC"]}
        plan = discover_tcga(selection, session=session)
        self.assertEqual(selection, {"projects": ["TCGA-SYNTHETIC"]})
        self.assertEqual(len(plan.assets), 2)
        asset = plan.assets[0]
        self.assertEqual(asset.subject_ids, ["case-a", "case-b"])
        self.assertEqual(asset.clinical_refs, ["case:case-a", "case:case-b"])
        self.assertEqual(asset.sample_ids, ["sample-case-a", "sample-case-b"])
        self.assertEqual(
            asset.slide_id, "file-a"
        )  # Multiple source slide entities remain uncollapsed.
        self.assertEqual(asset.relative_path, "slides/file-a.svs")
        self.assertEqual(asset.url, API_BASE + "/data/file-a")
        self.assertEqual(asset.checksums, {"md5": "a" * 32})
        self.assertEqual(asset.label_status, "unavailable")
        clinical = {
            record.record_id: record for record in plan.records if record.kind == "clinical"
        }
        self.assertEqual(clinical["case:case-a"].data, case_a)
        self.assertEqual(clinical["case:case-b"].subject_ids, ["case-b"])
        self.assertEqual(
            next(record for record in plan.records if record.record_id == "file:file-a").data, first
        )
        self.assertEqual(plan.provenance["release"], {"data_release": "synthetic-release"})
        self.assertEqual(plan.provenance["clinical_availability"], "available")
        self.assertEqual([call[2]["json"]["from"] for call in session.calls], [0, 1, 0, 1])
        file_request = session.calls[0][2]["json"]
        requested_fields = file_request["fields"].split(",")
        self.assertNotIn("associated_entities", requested_fields)
        self.assertIn("associated_entities.case_id", requested_fields)
        self.assertIn("associated_entities.entity_id", requested_fields)
        self.assertIn("associated_entities.entity_type", requested_fields)
        self.assertIn("associated_entities", file_request["expand"].split(","))
        filters = {
            condition["content"]["field"]: condition["content"]["value"]
            for condition in file_request["filters"]["content"]
        }
        self.assertEqual(filters["access"], ["open"])
        self.assertEqual(filters["experimental_strategy"], ["Diagnostic Slide"])
        case_request = session.calls[2][2]["json"]
        self.assertNotIn("fields", case_request)
        self.assertIn("follow_ups", case_request["expand"])
        self.assertEqual(session.posts, [])

    def test_exact_limit_explicit_ids_and_missing_clinical(self):
        session = Session(
            posts=[
                (
                    "files",
                    gdc([slide("file-a", ["case-a"]), slide("file-b", ["case-b"])], total=1000),
                ),
                ("cases", gdc([])),
            ]
        )
        plan = discover_tcga(
            {"file_ids": ["file-a"], "limit": 1, "diagnostic_only": False}, session=session
        )
        self.assertEqual([asset.asset_id for asset in plan.assets], ["file-a"])
        self.assertEqual(plan.assets[0].slide_id, "slide-file-a")
        self.assertEqual(plan.provenance["missing_clinical_case_ids"], ["case-a"])
        self.assertEqual(plan.provenance["clinical_availability"], "unavailable")
        request = session.calls[0][2]["json"]
        self.assertEqual(request["size"], 1)
        self.assertNotIn("return_type", request)
        filters = {
            item["content"]["field"]: item["content"]["value"]
            for item in request["filters"]["content"]
        }
        self.assertEqual(filters["file_id"], ["file-a"])
        self.assertNotIn("experimental_strategy", filters)

    def test_clinical_batching(self):
        subjects = [f"case-{index:03}" for index in range(101)]
        session = Session(
            posts=[
                ("files", gdc([slide("file-a", subjects)])),
                ("cases", gdc([{"case_id": subject} for subject in subjects[:100]])),
                ("cases", gdc([{"case_id": subjects[-1]}])),
            ]
        )
        plan = discover_tcga({"case_ids": subjects}, session=session)
        self.assertEqual(plan.provenance["clinical_case_count"], 101)
        self.assertEqual(len(session.calls[1][2]["json"]["filters"]["content"]["value"]), 100)
        self.assertEqual(len(session.calls[2][2]["json"]["filters"]["content"]["value"]), 1)

    def test_rejects_unsafe_selection_and_nonopen_or_duplicate_results(self):
        for selection in (
            {},
            {"projects": []},
            {"projects": ["OTHER"]},
            {"file_ids": ["../escape"]},
            {"file_ids": ["file-a"], "token": "secret"},
            {"file_ids": ["file-a"], "limit": True},
        ):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                discover_tcga(selection, session=Session())
        with self.assertRaisesRegex(ValueError, "open-access"):
            discover_tcga(
                {"file_ids": ["file-a"]},
                session=Session(posts=[("files", gdc([slide("file-a", [], access="controlled")]))]),
            )
        with self.assertRaisesRegex(ValueError, "duplicate file"):
            discover_tcga(
                {"file_ids": ["file-a"]},
                session=Session(posts=[("files", gdc([slide("file-a", []), slide("file-a", [])]))]),
            )


class CamelyonTests(unittest.TestCase):
    def test_native_flat_slides_reference_rows_and_checksum_source(self):
        session = cam_session()
        plan = discover_camelyon({"splits": ["all"]}, session=session)
        slides = {asset.slide_id: asset for asset in plan.assets if asset.kind == "slide"}
        self.assertEqual(set(slides), {"normal_001", "test_001", "tumor_001"})
        self.assertEqual(slides["normal_001"].split, "train")
        self.assertEqual(slides["test_001"].split, "test")
        self.assertEqual(slides["normal_001"].labels, {"type": "normal", "training_tumor": False})
        self.assertEqual(slides["test_001"].labels, {"type": "tumor"})
        self.assertEqual(slides["normal_001"].checksums, {"md5": "b" * 32})
        self.assertEqual(slides["test_001"].checksums, {})
        self.assertEqual(
            slides["tumor_001"].checksums, {}
        )  # Even a plain-looking ETag is not trusted as MD5.
        self.assertEqual(slides["test_001"].subject_ids, [])
        self.assertEqual(slides["test_001"].clinical_refs, [])
        self.assertEqual(slides["test_001"].relative_path, "slides/test_001.tif")
        self.assertIn("/CAMELYON16/images/test_001.tif", slides["test_001"].url)
        source_row = next(record for record in plan.records if record.record_id == "reference:1")
        self.assertEqual(
            source_row.data, {"image": "test_001.tif", "type": "tumor", "split": "test"}
        )
        self.assertEqual(plan.provenance["clinical_availability"], "unavailable")
        self.assertFalse(plan.provenance["etag_is_checksum"])
        self.assertEqual(sum(asset.kind == "license" for asset in plan.assets), 1)
        # Metadata-only discovery never GETs a slide or annotation payload URL.
        self.assertFalse(any(call[1].lower().endswith((".tif", ".xml")) for call in session.calls))

    def test_explicit_annotations_only_for_selected_slides(self):
        session = cam_session(annotations=True)
        plan = discover_camelyon(
            {"slide_ids": ["test_001.tif"], "include_annotations": True}, session=session
        )
        slides = [asset for asset in plan.assets if asset.kind == "slide"]
        self.assertEqual([asset.slide_id for asset in slides], ["test_001"])
        annotations = [asset for asset in plan.assets if asset.kind == "annotation"]
        self.assertEqual(
            {asset.filename for asset in annotations}, {"test_001.xml", "test_001_mask.tif"}
        )
        self.assertTrue(all(asset.related_slide_ids == ["test_001"] for asset in annotations))
        self.assertFalse(any(call[1].lower().endswith((".tif", ".xml")) for call in session.calls))

    def test_split_and_label_conflicts_remain_explicit(self):
        reference = "image,type,split\nnormal_001.tif,tumor,test\ntest_001.tif,normal,test\n"
        plan = discover_camelyon({"split": "train"}, session=cam_session(reference=reference))
        normal = next(asset for asset in plan.assets if asset.slide_id == "normal_001")
        self.assertEqual(normal.label_status, "conflicting")
        self.assertIsNone(normal.split)
        self.assertEqual(normal.labels, {"type": "tumor", "training_tumor": False})
        self.assertEqual(normal.metadata["native_split"], "train")
        self.assertEqual(normal.metadata["reference_split"], "test")
        self.assertTrue(normal.metadata["label_conflict"])

    def test_no_reference_label_inference_from_test_filename(self):
        plan = discover_camelyon(
            {"split": "test"},
            session=cam_session(reference="image,unrecognized\ntest_001.tif,value\n"),
        )
        test = next(asset for asset in plan.assets if asset.kind == "slide")
        self.assertEqual(test.labels, {})
        self.assertEqual(test.label_status, "missing")
        self.assertEqual(test.subject_ids, [])

    def test_missing_small_metadata_and_selection_validation(self):
        session = cam_session()
        session.gets[BASE_URL + "/" + PREFIX + "evaluation/reference.csv"] = Response(status=404)
        plan = discover_camelyon({"split": "test"}, session=session)
        test = next(asset for asset in plan.assets if asset.kind == "slide")
        self.assertEqual(test.label_status, "unavailable")
        self.assertEqual(plan.provenance["reference_availability"], "unavailable")
        for selection in (
            {},
            {"split": "validation"},
            {"slide_ids": ["../escape"]},
            {"split": "train", "include_annotations": "False"},
            {"split": "train", "token": "secret"},
        ):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                discover_camelyon(selection, session=Session())
        with self.assertRaisesRegex(ValueError, "not present"):
            discover_camelyon({"slide_ids": ["test_missing"]}, session=cam_session())

    def test_malformed_checksum_and_broken_listing_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "checksum row"):
            discover_camelyon({"split": "train"}, session=cam_session(checksum="not a checksum"))
        session = cam_session()
        session.gets[(BASE_URL, PREFIX + "images/", None)] = listing([], next_token="repeated")
        session.gets[(BASE_URL, PREFIX + "images/", "repeated")] = listing(
            [], next_token="repeated"
        )
        with self.assertRaisesRegex(ValueError, "continuation token"):
            discover_camelyon({"split": "train"}, session=session)


if __name__ == "__main__":
    unittest.main()
