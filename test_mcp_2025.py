#!/usr/bin/env python3
"""
Test NSAF MCP Server 2025 Protocol Compliance
==============================================

Tests the updated MCP server against 2025 JSON-RPC standards.
"""

import json
import subprocess
import sys
import time
import threading
from typing import Dict, Any
import signal
import os

class MCPServerTester:
    """Test the NSAF MCP Server for 2025 compliance"""
    
    def __init__(self):
        self.server_process = None
        self.test_results = []
        
    def start_server(self):
        """Start the MCP server process"""
        print("🚀 Starting NSAF MCP Server...")
        self.server_process = subprocess.Popen(
            [sys.executable, "nsaf_mcp_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        time.sleep(2)  # Give server time to start
        
    def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request and get response"""
        if not self.server_process:
            raise RuntimeError("Server not started")
            
        request_line = json.dumps(request) + '\n'
        self.server_process.stdin.write(request_line)
        self.server_process.stdin.flush()
        
        response_line = self.server_process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from server")
            
        return json.loads(response_line.strip())
        
    def test_initialization(self):
        """Test MCP 2025 initialization sequence"""
        print("\n📋 Testing MCP 2025 Initialization...")
        
        # Test initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {
                    "name": "NSAF Test Client",
                    "version": "1.0.0"
                }
            }
        }
        
        response = self.send_request(init_request)
        
        # Validate response
        assert "jsonrpc" in response and response["jsonrpc"] == "2.0"
        assert "id" in response and response["id"] == 1
        assert "result" in response
        assert "protocolVersion" in response["result"]
        assert "capabilities" in response["result"]
        assert "serverInfo" in response["result"]
        
        print("✅ Initialization test passed")
        return True
        
    def test_tools_list(self):
        """Test tools/list method"""
        print("\n📋 Testing tools/list method...")
        
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        response = self.send_request(request)
        
        # Validate response
        assert "jsonrpc" in response and response["jsonrpc"] == "2.0"
        assert "id" in response and response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]
        assert isinstance(response["result"]["tools"], list)
        assert len(response["result"]["tools"]) > 0
        
        # Check tool format
        tool = response["result"]["tools"][0]
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
        
        print(f"✅ Tools list test passed ({len(response['result']['tools'])} tools available)")
        return True
        
    def test_tool_call(self):
        """Test tools/call method"""
        print("\n📋 Testing tools/call method...")
        
        # Test get_nsaf_status tool
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_nsaf_status",
                "arguments": {
                    "include_details": False
                }
            }
        }
        
        response = self.send_request(request)
        
        # Validate response
        assert "jsonrpc" in response and response["jsonrpc"] == "2.0"
        assert "id" in response and response["id"] == 3
        assert "result" in response
        assert "content" in response["result"]
        assert isinstance(response["result"]["content"], list)
        
        print("✅ Tool call test passed")
        return True
        
    def test_error_handling(self):
        """Test error handling compliance"""
        print("\n📋 Testing JSON-RPC 2.0 error handling...")
        
        # Test invalid JSON-RPC version
        invalid_request = {
            "id": 4,
            "method": "tools/list"
            # Missing "jsonrpc": "2.0"
        }
        
        response = self.send_request(invalid_request)
        
        # Validate error response
        assert "jsonrpc" in response and response["jsonrpc"] == "2.0"
        assert "error" in response
        assert "code" in response["error"]
        assert response["error"]["code"] == -32600  # Invalid Request
        
        print("✅ Error handling test passed")
        return True
        
    def test_unknown_method(self):
        """Test unknown method handling"""
        print("\n📋 Testing unknown method handling...")
        
        request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "unknown/method"
        }
        
        response = self.send_request(request)
        
        # Validate error response
        assert "jsonrpc" in response and response["jsonrpc"] == "2.0"
        assert "error" in response
        assert response["error"]["code"] == -32601  # Method not found
        
        print("✅ Unknown method test passed")
        return True
        
    def stop_server(self):
        """Stop the MCP server"""
        if self.server_process:
            print("\n🛑 Stopping MCP Server...")
            self.server_process.terminate()
            self.server_process.wait()
            
    def run_all_tests(self):
        """Run all compliance tests"""
        print("🧪 NSAF MCP 2025 Protocol Compliance Test")
        print("=" * 50)
        
        try:
            self.start_server()
            
            # Run tests in sequence
            tests = [
                self.test_initialization,
                self.test_tools_list,
                self.test_tool_call,
                self.test_error_handling,
                self.test_unknown_method
            ]
            
            passed = 0
            for test in tests:
                try:
                    if test():
                        passed += 1
                except Exception as e:
                    print(f"❌ Test failed: {e}")
                    
            print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
            
            if passed == len(tests):
                print("🎉 All tests passed! MCP 2025 compliance verified.")
                return True
            else:
                print("⚠️ Some tests failed. Please review the implementation.")
                return False
                
        except Exception as e:
            print(f"💥 Test suite failed: {e}")
            return False
        finally:
            self.stop_server()

def main():
    """Main test runner"""
    tester = MCPServerTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()