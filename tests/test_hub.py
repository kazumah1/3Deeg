import hashlib,json,tempfile,unittest
from pathlib import Path
from n3d.hub import required_files,verify_files

class HubIntegrityTests(unittest.TestCase):
 def test_selects_only_requested_subject_and_shared_files(self):
  names=required_files(['sub02'],'point_cloud_simple')
  self.assertIn('packed/sub02_eeg.npz',names)
  self.assertIn('packed/objects.json',names)
  self.assertIn('packed/geometry_point_cloud_simple.npy',names)
  self.assertNotIn('packed/sub01_eeg.npz',names)
  self.assertEqual(len(names),5)
 def test_rejects_invalid_subject(self):
  for name in ['../../token','sub00','sub13','sub1']:
   with self.assertRaises(ValueError):required_files([name],'point_cloud_simple')
 def test_detects_same_size_tampering_and_missing_files(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'packed').mkdir();p=root/'packed/x';p.write_bytes(b'original')
   manifest={'files':{'packed/x':{'bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()}}}
   self.assertEqual(len(verify_files(root,manifest,['packed/x'])),1)
   p.write_bytes(b'tampered')
   with self.assertRaises(ValueError):verify_files(root,manifest,['packed/x'])
   p.unlink()
   with self.assertRaises((ValueError,FileNotFoundError)):verify_files(root,manifest,['packed/x'])
 def test_rejects_manifest_path_escape(self):
  with tempfile.TemporaryDirectory() as tmp:
   with self.assertRaises(ValueError):verify_files(Path(tmp),{'files':{'../outside':{'bytes':0,'sha256':''}}},['../outside'])

if __name__=='__main__':unittest.main()
