#!/usr/bin/env python3
"""
Test script for Triton-FastAPI integration
"""

import base64
import json
import requests
import sys
import time
import argparse


def test_health():
    """Test FastAPI health endpoint"""
    print("="*70)
    print("TESTING FASTAPI HEALTH ENDPOINT")
    print("="*70)
    
    try:
        response = requests.get("http://localhost:8005/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(json.dumps(data, indent=2))
        
        if data.get("speechlmm") == "loaded":
            print("\n✓ SpeechLMM engine is loaded")
            return True
        else:
            print("\n✗ SpeechLMM engine is NOT loaded")
            return False
    except Exception as e:
        print(f"\n✗ Health check failed: {e}")
        return False


def test_triton_direct(audio_file, prompt="Transcribe the following audio:"):
    """Test Triton server directly"""
    print("\n" + "="*70)
    print("TESTING TRITON SERVER DIRECTLY")
    print("="*70)
    print(f"Audio file: {audio_file}")
    print(f"Prompt: {prompt}")
    
    try:
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        file_size_mb = len(audio_bytes) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        payload = {
            "inputs": [
                {
                    "name": "AUDIO_BYTES",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [[audio_b64]]
                },
                {
                    "name": "TEXT_PROMPT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [[prompt]]
                }
            ],
            "outputs": [{"name": "GENERATED_TEXT"}]
        }
        
        print("\nSending request to Triton...")
        start = time.time()
        response = requests.post(
            "http://localhost:9000/v2/models/speechlmm/infer",
            json=payload,
            timeout=1800
        )
        elapsed = time.time() - start
        
        response.raise_for_status()
        result = response.json()
        output = result.get("outputs", [{}])[0]
        data = output.get("data", [])
        
        if data:
            text = data[0] if not isinstance(data[0], list) else data[0][0]
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            
            print(f"\n✓ Triton response received in {elapsed:.1f} seconds")
            print(f"\nTranscription (first 200 chars):")
            print("-" * 70)
            print(text[:200] + "..." if len(text) > 200 else text)
            print("-" * 70)
            print(f"Full length: {len(text)} characters")
            return True, text
        else:
            print("\n✗ No output data in response")
            print("Response:", json.dumps(result, indent=2)[:500])
            return False, None
            
    except Exception as e:
        print(f"\n✗ Triton test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_fastapi_transcribe(audio_file, prompt="Transcribe the following audio:"):
    """Test FastAPI transcription endpoint"""
    print("\n" + "="*70)
    print("TESTING FASTAPI TRANSCRIPTION ENDPOINT")
    print("="*70)
    print(f"Audio file: {audio_file}")
    print(f"Prompt: {prompt}")
    
    try:
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        file_size_mb = len(audio_bytes) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        payload = {
            "type": "full_transcript",
            "audio": audio_b64,
            "language": "en-US",
            "meeting_id": "test_meeting",
            "room_name": "test_room",
            "participant_id": "test_user"
        }
        
        print("\nSending request to FastAPI...")
        start = time.time()
        response = requests.post(
            "http://localhost:8005/api/transcribe",
            json=payload,
            timeout=1800,
            headers={"Content-Type": "application/json"}
        )
        elapsed = time.time() - start
        
        response.raise_for_status()
        result = response.json()
        
        if "transcript" in result:
            transcript = result["transcript"]
            print(f"\n✓ FastAPI response received in {elapsed:.1f} seconds")
            print(f"\nTranscription (first 200 chars):")
            print("-" * 70)
            print(transcript[:200] + "..." if len(transcript) > 200 else transcript)
            print("-" * 70)
            print(f"Full length: {len(transcript)} characters")
            print(f"\nFull response keys: {list(result.keys())}")
            return True, transcript
        else:
            print("\n✗ No transcript in response")
            print("Response:", json.dumps(result, indent=2)[:500])
            return False, None
            
    except Exception as e:
        print(f"\n✗ FastAPI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser(description="Test Triton-FastAPI integration")
    parser.add_argument("audio_file", help="Path to audio file (WAV format)")
    parser.add_argument("--prompt", "-p", default="Transcribe the following audio:",
                       help="Text prompt for transcription")
    parser.add_argument("--skip-triton", action="store_true",
                       help="Skip direct Triton test")
    parser.add_argument("--skip-fastapi", action="store_true",
                       help="Skip FastAPI test")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("TRITON-FASTAPI INTEGRATION TEST")
    print("="*70)
    
    results = {}
    
    # Test 1: Health check
    results["health"] = test_health()
    
    if not results["health"]:
        print("\n✗ Health check failed. Please check FastAPI server logs.")
        sys.exit(1)
    
    # Test 2: Direct Triton test
    if not args.skip_triton:
        success, triton_text = test_triton_direct(args.audio_file, args.prompt)
        results["triton"] = success
        results["triton_text"] = triton_text
    else:
        print("\nSkipping direct Triton test (--skip-triton)")
        results["triton"] = None
    
    # Test 3: FastAPI endpoint test
    if not args.skip_fastapi:
        success, fastapi_text = test_fastapi_transcribe(args.audio_file, args.prompt)
        results["fastapi"] = success
        results["fastapi_text"] = fastapi_text
    else:
        print("\nSkipping FastAPI test (--skip-fastapi)")
        results["fastapi"] = None
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Health check: {'✓ PASS' if results['health'] else '✗ FAIL'}")
    if results.get("triton") is not None:
        print(f"Direct Triton: {'✓ PASS' if results['triton'] else '✗ FAIL'}")
    if results.get("fastapi") is not None:
        print(f"FastAPI endpoint: {'✓ PASS' if results['fastapi'] else '✗ FAIL'}")
    
    if results.get("triton_text") and results.get("fastapi_text"):
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        triton_len = len(results["triton_text"])
        fastapi_len = len(results["fastapi_text"])
        print(f"Triton direct length: {triton_len} chars")
        print(f"FastAPI length: {fastapi_len} chars")
        
        if triton_len > 0 and fastapi_len > 0:
            similarity = min(triton_len, fastapi_len) / max(triton_len, fastapi_len)
            print(f"Length similarity: {similarity*100:.1f}%")
    
    print("\n" + "="*70)
    
    if all(v for k, v in results.items() if k in ["health", "triton", "fastapi"] and v is not None):
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

