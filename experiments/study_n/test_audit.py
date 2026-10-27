"""Full synthetic replay and counterexamples to accepting altered evidence."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.study_n import audit, run
from experiments.study_n.tasks import make_world


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.dev, cls.formal = cls.base / 'development', cls.base / 'formal'
        cls.data, cls.manifest = cls.base / 'worlds.json', cls.base / 'manifest.json'
        run.document(cls.data, [make_world(-273), make_world(-274)])
        run.document(cls.manifest, dict(source_sha256=run.source_hashes(), source_revision='synthetic',
            data_sha256=hashlib.sha256(cls.data.read_bytes()).hexdigest()))
        cls.patches = ExitStack()
        for module in (run, audit):
            for name, value in dict(DATA=cls.data, MANIFEST=cls.manifest,
                                    DEVELOPMENT_SEED=-273, FORMAL_SEEDS=(-274,)).items():
                cls.patches.enter_context(patch.object(module, name, value))
        with patch.object(sys, 'argv', ['run', '--development', '--output', str(cls.dev)]):
            run.main()
        certificate, _ = audit.audit_run(cls.dev)
        cls.certificate = cls.base / 'audit.json'
        run.document(cls.certificate, certificate)
        with patch.object(sys, 'argv', ['run', '--output', str(cls.formal),
                                      '--development-audit', str(cls.certificate)]):
            run.main()
        cls.folder = cls.dev / 'seed--273'

    @classmethod
    def tearDownClass(cls):
        cls.patches.close()
        cls.temporary.cleanup()

    def test_complete_development_and_formal_replay(self):
        for root in (self.dev, self.formal):
            report, phases = audit.audit_run(root)
            self.assertEqual(report['counts'], dict(queries=64, gradients=1280,
                checkpoints=16, direct=20, probes=27776))
            self.assertEqual(len([p for p in phases if p['stage'] == 'training']), 256)
            self.assertEqual(report['files'], audit.file_hashes(root))

    def test_changed_queries_gradients_predictions_and_evaluations_fail(self):
        for journal, field, value in [('queries', 'next_state', 99), ('updates', 'loss', -1.),
                ('predictions', 'prediction', 99), ('graphs', 'logits_sha256', 'altered'),
                ('direct', 'exact', -1), ('probes', 'success', None), ('reference', 'exact', -1)]:
            path = self.folder / (journal + '.jsonl')
            original = path.read_bytes()
            try:
                rows = list(audit.records(path))
                rows[0][field] = value
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                with self.subTest(journal=journal), self.assertRaises(AssertionError):
                    audit.audit_run(self.dev)
            finally:
                path.write_bytes(original)

    def test_altered_weights_adam_and_tensor_dtype_fail(self):
        path = self.folder / 'canonical/small/16.pt'
        original = path.read_bytes()
        for target in ('model', 'optimizer', 'dtype'):
            try:
                saved = torch.load(path, weights_only=True)
                if target == 'model':
                    saved['model']['input_layer.weight'][0, 0] += .01
                elif target == 'optimizer':
                    saved['optimizer']['state'][0]['exp_avg'][0, 0] += .01
                else:
                    saved['model']['input_layer.weight'] = saved['model']['input_layer.weight'].to(torch.complex128)
                torch.save(saved, path)
                with self.subTest(target=target), self.assertRaises(AssertionError):
                    audit.audit_run(self.dev)
            finally:
                path.write_bytes(original)

    def test_extra_null_row_and_truncated_journal_fail(self):
        path = self.folder / 'queries.jsonl'
        original = path.read_bytes()
        for raw in (original + b'null\n{}\n', original[:original.rfind(b'\n', 0, -1) + 1]):
            try:
                path.write_bytes(raw)
                with self.assertRaises((AssertionError, StopIteration)):
                    audit.audit_run(self.dev)
            finally:
                path.write_bytes(original)

    def test_bad_timing_and_work_counts_fail(self):
        path = self.folder / 'phases.jsonl'
        original = path.read_bytes()
        for field, value in [('seconds', -1.), ('journal_seconds', 100000.), ('gradient_steps', 0)]:
            try:
                rows = list(audit.records(path))
                row = next(r for r in rows if r['stage'] == 'training')
                row[field] = value
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    audit.audit_run(self.dev)
            finally:
                path.write_bytes(original)

    def test_metadata_and_frozen_provenance_fail_when_changed(self):
        for path, field, value in [(self.dev / 'metadata.json', 'data_sha256', '0' * 64),
                (self.dev / 'metadata.json', 'source_revision', 'other'),
                (self.dev / 'metadata.json', 'elapsed_seconds', 0.),
                (self.manifest, 'source_sha256', {}), (self.manifest, 'data_sha256', '0' * 64)]:
            original = path.read_bytes()
            try:
                record = json.loads(original)
                record[field] = value
                run.document(path, record)
                with self.subTest(path=path, field=field), self.assertRaises(AssertionError):
                    audit.audit_run(self.dev)
            finally:
                path.write_bytes(original)

    def test_missing_extra_and_mid_audit_changed_files_fail(self):
        path = self.folder / 'canonical/small/0.pt'
        original = path.read_bytes()
        try:
            path.unlink()
            with self.assertRaises(AssertionError):
                audit.audit_run(self.dev)
        finally:
            path.write_bytes(original)
        extra = self.dev / 'unexpected.json'
        try:
            extra.write_text('{}\n')
            with self.assertRaises(AssertionError):
                audit.audit_run(self.dev)
        finally:
            extra.unlink()
        metadata = self.dev / 'metadata.json'
        original_metadata = metadata.read_bytes()
        read = audit.read_document
        def read_then_replace(path):
            value = read(path)
            if path == metadata:
                run.document(path, dict(value, status='failed'))
            return value
        try:
            with patch.object(audit, 'read_document', side_effect=read_then_replace), self.assertRaises(AssertionError):
                audit.audit_run(self.dev)
        finally:
            metadata.write_bytes(original_metadata)

    def test_incomplete_archived_development_certificate_fails_even_with_new_digest(self):
        path, metadata_path = self.formal / 'development-audit.json', self.formal / 'metadata.json'
        original, original_metadata = path.read_bytes(), metadata_path.read_bytes()
        for field, value in [('files', {}), ('counts', {}), ('source_revision', 'other')]:
            try:
                certificate = json.loads(original)
                certificate[field] = value
                run.document(path, certificate)
                metadata = json.loads(original_metadata)
                metadata['development_audit_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                run.document(metadata_path, metadata)
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    audit.audit_run(self.formal)
            finally:
                path.write_bytes(original)
                metadata_path.write_bytes(original_metadata)

    def test_optimized_python_is_refused(self):
        result = subprocess.run([sys.executable, '-O', '-c',
            "from pathlib import Path; from experiments.study_n.audit import audit_run; audit_run(Path('absent'))"],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Optimized Python disables audit assertions', result.stderr)


if __name__ == '__main__':
    unittest.main()
