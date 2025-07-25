from inference.pipeline import run_rag_pipeline

docs = [
    "Bangla: বাংলাদেশের স্বাধীনতা যুদ্ধ ১৯৭১ সালে সংঘটিত হয়।",
    "English: The capital of France is Paris."
]
query = "What is the capital of France?"

result = run_rag_pipeline(docs, query, lang="english")
print("🔍 Final Answer:", result)
