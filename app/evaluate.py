import requests

test_cases = [
    {"question": "How many sick days do I get?", "expected": "10"},
    {"question": "How many days can I work remotely?", "expected": "3"},
    {
        "question": "How much is the maximum deduction for lost equipment?",
        "expected": "200",
    },
]
passed_counts = []

for case in test_cases:
    response = requests.post(
        "http://127.0.0.1:8000/ask", json={"text": case["question"]}
    )
    result = response.json()
    answer = result["answer"]

    passed = case["expected"] in answer
    passed_counts.append(passed)

    print(f"Question: {case['question']}")
    print(f"Answer: {answer}")
    print(f"Expected to contain: {case['expected']}")
    print(f"PASSED: {passed}")
    print("---")


count = passed_counts.count(True)
print(f"Passed: {count} out of {len(test_cases)}")
