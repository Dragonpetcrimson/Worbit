import unittest
from unittest import mock
from gpt_summarizer import generate_summary_from_clusters

class TestGPTSummarizer(unittest.TestCase):
    def setUp(self):
        self.test_clusters = {
            0: [{'text': 'Test error 1', 'severity': 'High', 'file': 'test.log', 'line_num': 1}],
            1: [{'text': 'Test error 2', 'severity': 'Medium', 'file': 'test.log', 'line_num': 2}]
        }
        self.test_id = "TEST-MOCK-123"

    @mock.patch('gpt_summarizer.send_to_openai_chat')
    def test_gpt_call_with_mock(self, mock_send):
        # Simulate GPT returning a structured summary
        mock_send.return_value = (
            "1. ROOT CAUSE: Mock test response\n"
            "2. IMPACT: This is a mock impact section\n"
            "3. RECOMMENDED ACTIONS:\n- Use mocks in test\n- Validate behavior"
        )

        # Call function (it will use the mocked GPT call)
        result = generate_summary_from_clusters(
            self.test_clusters, [], self.test_id, use_gpt=True, model="gpt-4"
        )

        self.assertIn("ROOT CAUSE", result)
        self.assertIn("Mock test response", result)
        self.assertIn("RECOMMENDED ACTIONS", result)

        mock_send.assert_called_once()
        call_args = mock_send.call_args[0]
        self.assertIn("Test ID", call_args[0])  # The prompt text
        self.assertEqual(call_args[1], "gpt-4")  # The model

        print("✅ GPT mock test passed")

if __name__ == "__main__":
    unittest.main()
