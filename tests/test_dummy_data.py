"""No model dependencies or GPU required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('dummy_dataset',Path(__file__).resolve().parents[1]/'scripts/dataset.py')
dataset=importlib.util.module_from_spec(spec);spec.loader.exec_module(dataset)
class DummySafety(unittest.TestCase):
    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'dummy';dataset.dummy(root)
            before={str(p.relative_to(root)):(p.read_bytes(),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            dataset.dummy(root)
            after={str(p.relative_to(root)):(p.read_bytes(),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
    def test_foreign_folder_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'important.txt').write_text('private')
            with self.assertRaises(ValueError):dataset.dummy(root)
            self.assertEqual([p.name for p in root.iterdir()],['important.txt'])
            self.assertEqual((root/'important.txt').read_text(),'private')
    def test_modified_fixture_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'dummy';dataset.dummy(root)
            changed=root/'train/dummy_00.txt';changed.write_text('MODIFIED')
            before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with self.assertRaises(ValueError):dataset.dummy(root)
            self.assertEqual(before,{str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()})
if __name__=='__main__':unittest.main()
