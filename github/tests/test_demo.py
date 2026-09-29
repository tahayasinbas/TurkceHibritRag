"""The default demo must never invoke the paid generation step."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
import unittest
from unittest.mock import patch

from turkish_rag.__main__ import main


class DemoCommandTests(unittest.TestCase):
    def run_demo(self, *arguments):
        with patch('sys.argv', ['turkish_rag', 'demo', *arguments]), \
             patch('turkish_rag.pipeline.ingest') as ingest, \
             patch('turkish_rag.pipeline.RAG') as rag_class, \
             patch('turkish_rag.store.connect'), \
             redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()):
            rag = rag_class.return_value
            rag.retrieve.return_value = [{'id': 'example-source'}]
            rag.answer.return_value = {'answer': 'example response'}
            main()
            return json.loads(stdout.getvalue()), ingest, rag

    def test_default_only_retrieves(self):
        result, ingest, rag = self.run_demo('Hibrit erişim nedir?')
        ingest.assert_called_once()
        rag.retrieve.assert_called_once_with('Hibrit erişim nedir?')
        rag.answer.assert_not_called()
        self.assertTrue(result['demo'])
        self.assertEqual(result['sources'], [{'id': 'example-source'}])

    def test_generation_requires_explicit_flag(self):
        result, _, rag = self.run_demo('Kaynakları özetle.', '--generate', '--model', 'example-model')
        rag.answer.assert_called_once_with('Kaynakları özetle.', 'example-model')
        self.assertEqual(result['answer'], 'example response')


if __name__ == '__main__':
    unittest.main()
