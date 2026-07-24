"""
Script to test inference logic with the updated category-aware clustering locally.
"""

import json
import os
import sys

# Mock Lambda artifacts directory for local testing
lambda_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lambda"))
if lambda_path not in sys.path:
    sys.path.append(lambda_path)

import handler

def test_inference():
    print("Testing category-aware inference fallback behavior...")
    
    # 1. Single category test (fallback expected if no per_category set, else per_category)
    event_single = {
        "texts": [
            "the shipping was terrible and the box arrived crushed",
            "the app keeps crashing when I try to login"
        ],
        "categories": [
            "Electronics",
            "Software"
        ]
    }
    
    print("\n[Input]")
    print(json.dumps(event_single, indent=2))
    
    response = handler.lambda_handler(event_single, None)
    results = json.loads(response["body"])["results"]
    
    print("\n[Output]")
    for r in results:
        print(f"Text: {r['text'][:30]}...")
        print(f"  Sentiment: {r['sentiment']}")
        if r['sentiment'] == 'negative':
            print(f"  Issue Tag: {r['issue_tag']}")
            print(f"  Distance: {r['issue_distance']:.3f}")
            print(f"  Cluster Source: {r.get('cluster_source', 'N/A')}")
            
    print("\n" + "-"*50 + "\n")
    
    # 2. Multiple categories and missing categories
    event_multiple = {
        "texts": [
            "completely stopped working after a week",
            "tastes awful, dog won't eat it",
            "it is okay I guess"
        ],
        "categories": [
            "Unknown Category", # Should use fallback
            "Pet Supplies",     # Might use per_category if we have one
            ""                  # Missing category should use fallback
        ]
    }
    
    print("\n[Input]")
    print(json.dumps(event_multiple, indent=2))
    
    response = handler.lambda_handler(event_multiple, None)
    results = json.loads(response["body"])["results"]
    
    print("\n[Output]")
    for r in results:
        print(f"Text: {r['text'][:30]}...")
        print(f"  Sentiment: {r['sentiment']}")
        if r['sentiment'] == 'negative':
            print(f"  Issue Tag: {r['issue_tag']}")
            print(f"  Cluster Source: {r.get('cluster_source', 'N/A')}")
            
if __name__ == "__main__":
    test_inference()
