"""Archive tests use generated RGB images and temporary paths only."""

import multiprocessing
from pathlib import Path
import pickle
import tempfile
import unittest

import h5py
import numpy as np

from openhisto.patch_extraction.archive import PatchArchiveWriter, PatchReader


def _read_in_worker(reader, queue):
    try:
        record = reader[-1]
        queue.put((record.index, record.xy_level0, int(record.image[0, 0, 0])))
        reader.close()
    except Exception as exc:
        queue.put(("error", repr(exc)))


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "patches.h5"
        self.image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

    def write_archive(self, *, codec="png", count=3):
        with PatchArchiveWriter(
            self.path,
            patch_size=8,
            metadata={"target_mag": 20.0, "settings": {"edge": "pad"}},
            codec=codec,
        ) as writer:
            self.assertFalse(self.path.exists())
            for index in range(count):
                image = self.image if index == 0 else np.full_like(self.image, index * 20)
                self.assertEqual(
                    writer.append(image, (index * 16, 4), (16, 16), (0, 0, 8, 7)), index
                )
            writer.set_counts({"candidate_count": count + 1, "patch_count": count})
            self.assertEqual(len(writer), count)

    def test_png_roundtrip_and_storage_layout(self):
        self.write_archive()
        with h5py.File(self.path, "r") as handle:
            self.assertIsNone(handle["images/bytes"].compression)
            self.assertLessEqual(np.prod(handle["images/bytes"].chunks), 64 * 1024)
            self.assertEqual(handle["coords"].chunks, (256, 2))
            self.assertEqual(handle["images/offsets"].shape, (4,))
            self.assertEqual(handle["coords"].dtype, np.dtype("int64"))
            self.assertEqual(handle["valid_bounds"].dtype, np.dtype("int32"))
        with PatchReader(self.path) as reader:
            self.assertEqual(len(reader), 3)
            np.testing.assert_array_equal(reader[0].image, self.image)
            self.assertEqual(reader[-1].index, 2)
            self.assertEqual(reader[-1].xy_level0, (32, 4))
            self.assertEqual(reader[0].footprint_level0, (16, 16))
            self.assertEqual(reader[0].valid_bounds, (0, 0, 8, 7))
            self.assertEqual([record.index for record in reader], [0, 1, 2])
            np.testing.assert_array_equal(reader.get_coordinates(1), [[16, 4], [32, 4]])
            np.testing.assert_array_equal(reader.get_coordinates(-1), [[32, 4]])
            self.assertEqual(reader.get_coordinates(2, 1).shape, (0, 2))
            self.assertEqual(reader.metadata["counts"], {"candidate_count": 4, "patch_count": 3})
            self.assertTrue(reader.metadata["complete"])
            metadata = reader.metadata
            metadata["settings"]["edge"] = "changed"
            self.assertEqual(reader.metadata["settings"]["edge"], "pad")
            with self.assertRaises(IndexError):
                reader[3]
            with self.assertRaises(IndexError):
                reader[-4]
            with self.assertRaises(TypeError):
                reader[0.5]
        self.assertIsNone(reader._file)
        self.assertEqual(len(reader), 3)
        reader.close()

    def test_jpeg_roundtrip(self):
        self.write_archive(codec="jpeg")
        with PatchReader(self.path) as reader:
            self.assertEqual(reader.metadata["codec"], "jpeg")
            np.testing.assert_allclose(reader[1].image, np.full_like(self.image, 20), atol=2)
            self.assertEqual(reader[0].image.shape, (8, 8, 3))
            self.assertEqual(reader[0].image.dtype, np.uint8)

    def test_empty_archive(self):
        self.write_archive(count=0)
        with PatchReader(self.path) as reader:
            self.assertEqual(len(reader), 0)
            self.assertEqual(list(reader), [])
            self.assertEqual(reader.get_coordinates().shape, (0, 2))
            with self.assertRaises(IndexError):
                reader[0]

    def test_error_cleans_temporary_and_preserves_existing(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with PatchArchiveWriter(self.path, patch_size=8, metadata={}) as writer:
                writer.append(self.image, (0, 0), (8, 8), (0, 0, 8, 8))
                raise RuntimeError("synthetic failure")
        self.assertEqual(list(self.root.iterdir()), [])
        self.write_archive()
        original = self.path.read_bytes()
        with self.assertRaises(RuntimeError):
            with PatchArchiveWriter(self.path, patch_size=8, metadata={}, overwrite=True):
                raise RuntimeError("synthetic replacement failure")
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_overwrite_and_publication_race(self):
        self.write_archive()
        with self.assertRaises(FileExistsError):
            with PatchArchiveWriter(self.path, patch_size=8, metadata={}):
                pass
        with PatchArchiveWriter(self.path, patch_size=8, metadata={}, overwrite=True):
            pass
        with PatchReader(self.path) as reader:
            self.assertEqual(len(reader), 0)
        self.path.unlink()
        with self.assertRaises(FileExistsError):
            with PatchArchiveWriter(self.path, patch_size=8, metadata={}):
                self.path.write_bytes(b"concurrent writer output")
        self.assertEqual(self.path.read_bytes(), b"concurrent writer output")
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_invalid_inputs_do_not_leave_archive(self):
        with self.assertRaisesRegex(ValueError, "overwrite must be a boolean"):
            PatchArchiveWriter(self.path, patch_size=8, metadata={}, overwrite="False")
        with self.assertRaises(ValueError):
            PatchArchiveWriter(self.path, patch_size=8, metadata={"value": float("nan")})
        with self.assertRaises(ValueError):
            with PatchArchiveWriter(self.path, patch_size=8, metadata={}) as writer:
                writer.append(self.image.astype(np.float32), (0, 0), (8, 8), (0, 0, 8, 8))
        with self.assertRaises(ValueError):
            with PatchArchiveWriter(self.path, patch_size=8, metadata={}) as writer:
                writer.append(self.image, (0, 0), (8, 8), (0, 0, 9, 8))
        with self.assertRaises(ValueError):
            with PatchArchiveWriter(self.path, patch_size=8, metadata={}) as writer:
                writer.set_counts({"patch_count": 5})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_rejects_incomplete_schema_offsets_and_row_lengths(self):
        cases = (
            ("incomplete", lambda f: f.attrs.__setitem__("complete", False), "incomplete"),
            ("schema", lambda f: f.attrs.__setitem__("schema_version", 99), "schema_version"),
            ("offset_count", lambda f: f["images/offsets"].resize((2,)), "N \\+ 1"),
            ("offset_start", lambda f: f["images/offsets"].__setitem__(0, 1), "start at zero"),
            ("offset_order", lambda f: f["images/offsets"].__setitem__(1, 0), "increase strictly"),
            ("offset_end", lambda f: f["images/offsets"].__setitem__(-1, 0), "final offset"),
            ("row_count", lambda f: f["footprints"].resize((1, 2)), "row count"),
            ("missing_dataset", lambda f: f.__delitem__("coords"), "missing dataset"),
        )
        for name, corrupt, message in cases:
            with self.subTest(name=name):
                if self.path.exists():
                    self.path.unlink()
                self.write_archive()
                with h5py.File(self.path, "r+") as handle:
                    corrupt(handle)
                reader = PatchReader(self.path)
                with self.assertRaisesRegex(ValueError, message):
                    len(reader)
                self.assertIsNone(reader._file)

    def test_payload_corruption_is_reported_when_read(self):
        self.write_archive()
        with h5py.File(self.path, "r+") as handle:
            handle["images/bytes"][:8] = 0
        with PatchReader(self.path) as reader:
            self.assertEqual(len(reader), 3)
            with self.assertRaisesRegex(ValueError, "index 0"):
                reader[0]

    def test_pickling_drops_live_handle_and_spawn_worker_reads(self):
        self.write_archive()
        reader = PatchReader(self.path)
        self.addCleanup(reader.close)
        self.assertIsNone(reader._file)
        self.assertEqual(len(reader), 3)
        restored = pickle.loads(pickle.dumps(reader))
        self.addCleanup(restored.close)
        self.assertIsNone(restored._file)
        self.assertIsNotNone(reader._file)
        self.assertEqual(restored[-1].xy_level0, (32, 4))
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        self.addCleanup(queue.close)
        worker = context.Process(target=_read_in_worker, args=(reader, queue))
        worker.start()
        worker.join(timeout=20)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
            self.fail("Spawn worker did not finish")
        self.assertEqual(worker.exitcode, 0)
        self.assertEqual(queue.get(timeout=5), (2, (32, 4), 40))
        self.assertEqual(reader[0].index, 0)

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "fork is unavailable")
    def test_fork_worker_reopens_parent_handle(self):
        self.write_archive()
        reader = PatchReader(self.path)
        self.addCleanup(reader.close)
        self.assertEqual(len(reader), 3)
        context = multiprocessing.get_context("fork")
        queue = context.Queue()
        self.addCleanup(queue.close)
        worker = context.Process(target=_read_in_worker, args=(reader, queue))
        worker.start()
        worker.join(timeout=20)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
            self.fail("Fork worker did not finish")
        self.assertEqual(worker.exitcode, 0)
        self.assertEqual(queue.get(timeout=5), (2, (32, 4), 40))
        self.assertEqual(reader[0].index, 0)


if __name__ == "__main__":
    unittest.main()
