#!/usr/bin/env python3
"""
Test all FastAPI tasks with Triton integration:
1. Transcription
2. Captioning
3. Summarization
"""

import base64
import requests
import json
import sys
import time


def test_transcription(audio_file):
    """Test transcription endpoint"""
    print("\n" + "="*70)
    print("TEST 1: TRANSCRIPTION")
    print("="*70)
    
    with open(audio_file, 'rb') as f:
        audio_bytes = f.read()
    
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    payload = {
        'type': 'full_transcript',
        'audio': audio_b64,
        'language': 'en-US',
        'meeting_id': 'test_transcription',
        'room_name': 'test_room',
        'participant_id': 'test_user'
    }
    
    print(f"Audio size: {len(audio_bytes)/1024:.2f} KB")
    print("Sending request...")
    
    try:
        start = time.time()
        response = requests.post(
            'http://localhost:8005/api/transcribe',
            json=payload,
            timeout=120,
            headers={'Content-Type': 'application/json'}
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            transcript = result.get('transcript', '')
            print(f"✓ SUCCESS ({elapsed:.1f}s)")
            print(f"Transcription: {transcript[:150]}...")
            print(f"Length: {len(transcript)} chars")
            return True, transcript
        else:
            print(f"✗ FAILED: {response.status_code}")
            print(response.text[:500])
            return False, None
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False, None


def test_captioning(audio_file):
    """Test captioning endpoint"""
    print("\n" + "="*70)
    print("TEST 2: CAPTIONING")
    print("="*70)
    
    with open(audio_file, 'rb') as f:
        audio_bytes = f.read()
    
    print(f"Audio size: {len(audio_bytes)/1024:.2f} KB")
    print("Sending request...")
    
    try:
        start = time.time()
        files = {'audio': ('test.wav', audio_bytes, 'audio/wav')}
        data = {
            'language': 'en-US',
            'participant_id': 'test_user',
            'session_id': 'test_session'
        }
        
        response = requests.post(
            'http://localhost:8005/api/caption',
            files=files,
            data=data,
            timeout=120
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            caption = result.get('transcript', '')
            print(f"✓ SUCCESS ({elapsed:.1f}s)")
            print(f"Caption: {caption[:150]}...")
            print(f"Length: {len(caption)} chars")
            print(f"Is interim: {result.get('is_interim', 'N/A')}")
            return True, caption
        else:
            print(f"✗ FAILED: {response.status_code}")
            print(response.text[:500])
            return False, None
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_summarization(audio_file):
    """Test summarization endpoint"""
    print("\n" + "="*70)
    print("TEST 3: SUMMARIZATION")
    print("="*70)
    
    with open(audio_file, 'rb') as f:
        audio_bytes = f.read()
    
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    payload = {
        'type': 'summary',
        'audio': audio_b64,
        'language': 'en-US',
        'meeting_id': 'test_summary',
        'room_name': 'test_room',
        'participant_id': 'test_user'
    }
    
    print(f"Audio size: {len(audio_bytes)/1024:.2f} KB")
    print("Sending request...")
    
    try:
        start = time.time()
        response = requests.post(
            'http://localhost:8005/api/summarize',
            json=payload,
            timeout=120,
            headers={'Content-Type': 'application/json'}
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            summary = result.get('result', '')
            print(f"✓ SUCCESS ({elapsed:.1f}s)")
            print(f"Summary: {summary[:200]}...")
            print(f"Length: {len(summary)} chars")
            return True, summary
        else:
            print(f"✗ FAILED: {response.status_code}")
            print(response.text[:500])
            return False, None
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_all_tasks.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print("="*70)
    print("TESTING ALL FASTAPI TASKS WITH TRITON")
    print("="*70)
    print(f"Audio file: {audio_file}")
    
    results = {}
    
    results['transcription'] = test_transcription(audio_file)
    results['captioning'] = test_captioning(audio_file)
    results['summarization'] = test_summarization(audio_file)
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for task, (success, _) in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{task.capitalize()}: {status}")
    
    all_passed = all(success for success, _ in results.values())
    
    if all_passed:
        print("\n✓ ALL TASKS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME TASKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

