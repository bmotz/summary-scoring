import requests
import json

test_data = {
    "context": "The Industrial Revolution was a period of major changes in the way products were made. It began in the middle 1700s and lasted into the early 1900s. The Industrial Revolution began in Great Britain and soon spread to the United States and other countries. The main feature of the Industrial Revolution was the use of machines in factories for manufacturing goods. Before the Industrial Revolution, people made products by hand at home or in small workshops. During the Industrial Revolution, many factories were built, and they used machines to make products very quickly and efficiently.",
    "question": "Summarize the main points of this passage about the Industrial Revolution.",
    "student_response": "The Industrial Revolution was a significant period of change in manufacturing that occurred from the mid-1700s to early 1900s. Starting in Great Britain and spreading to other countries including the United States, it marked a shift from hand-made production in homes to machine-based manufacturing in factories. This new approach allowed for faster and more efficient production of goods."
}

response = requests.post(
    'http://localhost/llmscoring/summary',
    json=test_data,
    headers={'Content-Type': 'application/json'}
)

print(json.dumps(response.json(), indent=2))