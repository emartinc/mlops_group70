#!/bin/bash
# Script to test the MBTI API

API_URL="http://localhost:8000"

echo "Testing MBTI Personality Classifier API"
echo "========================================"
echo ""

# Test 1: Health check
echo "1. Health Check:"
curl -s -X GET "$API_URL/health" | python -m json.tool
echo ""
echo ""

# Test 2: Root endpoint
echo "2. Root Endpoint:"
curl -s -X GET "$API_URL/" | python -m json.tool
echo ""
echo ""

# Test 3: Prediction
echo "3. Prediction Test:"
echo "Testing with sample text..."

SAMPLE_TEXT="I love spending time alone with my thoughts. I enjoy analyzing complex problems and coming up with creative solutions. I prefer to plan things ahead rather than being spontaneous. When making decisions, I rely more on logic than emotions. I find deep conversations more interesting than small talk. I like to understand how things work and why they happen. I'm naturally skeptical and question assumptions. I value competence and intelligence in myself and others."

curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$SAMPLE_TEXT\"}" | python -m json.tool

echo ""
echo ""
echo "========================================"
echo "API Test Complete!"
