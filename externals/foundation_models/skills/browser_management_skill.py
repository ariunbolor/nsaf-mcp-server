from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import psutil
from .base_skill import BaseSkill
from .rag_skill import RAGSkill

class BrowserManagementSkill(BaseSkill):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="browser_management",
            description="A skill for managing browser operations including tabs, cache, bookmarks, and browser settings",
            config=config
        )
        self.required_credentials = [
            "browser_extension_api_key"  # For browser extension operations
        ]
        self.rag_skill = RAGSkill()
        self.metrics = {
            "tabs_opened": 0,
            "tabs_closed": 0,
            "cache_cleared_count": 0,
            "cookies_cleared_count": 0,
            "bookmarks_added": 0,
            "bookmarks_removed": 0,
            "successful_logins": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "passwords_saved": 0,
            "speed_tests_run": 0,
            "average_speed": 0,  # in Mbps
            "ads_skipped": 0,
            "links_copied": 0,
            "autofills_updated": 0,
            "tab_metrics": {
                "max_tabs_open": 0,
                "avg_tab_lifetime": 0,  # in minutes
                "most_visited_domains": {},
                "tab_categories": {
                    "social": 0,
                    "work": 0,
                    "shopping": 0,
                    "entertainment": 0,
                    "other": 0
                }
            },
            "bookmark_metrics": {
                "folder_distribution": {},
                "most_bookmarked_domains": {},
                "bookmark_categories": {
                    "news": 0,
                    "reference": 0,
                    "tools": 0,
                    "social": 0,
                    "other": 0
                },
                "unused_bookmarks": 0
            },
            "performance_metrics": {
                "avg_page_load_time": 0.0,  # in seconds
                "memory_usage": 0.0,  # in MB
                "cpu_usage": 0.0,  # percentage
                "bandwidth_usage": 0.0,  # in MB/s
                "cache_hit_rate": 0.0,  # percentage
                "cache_size": 0  # in MB
            },
            "security_metrics": {
                "logins_protected": 0,
                "passwords_encrypted": 0,
                "suspicious_sites_blocked": 0,
                "cookie_policies_enforced": 0,
                "secure_connections": 0,
                "insecure_connections": 0
            },
            "speed_test_metrics": {
                "peak_download": 0.0,  # in Mbps
                "peak_upload": 0.0,    # in Mbps
                "lowest_ping": 999,    # in ms
                "tests_by_server": {},
                "time_of_day_performance": {
                    "morning": 0.0,
                    "afternoon": 0.0,
                    "evening": 0.0,
                    "night": 0.0
                }
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters for browser operations"""
        errors = []
        required_params = {
            "action": ["open_tab", "close_tab", "clear_cache", "clear_cookies", 
                      "login", "logout", "save_password", "update_autofill",
                      "add_bookmark", "remove_bookmark", "check_speed",
                      "skip_ad", "copy_link"],
            "url": ["open_tab", "login", "add_bookmark"],
            "tab_id": ["close_tab"],
            "credentials": ["login", "save_password"],
            "autofill_data": ["update_autofill"],
            "bookmark_data": ["add_bookmark"],
            "bookmark_id": ["remove_bookmark"]
        }

        if "action" not in params:
            errors.append("Action parameter is required")
            return errors

        action = params["action"]
        if action not in required_params["action"]:
            errors.append(f"Invalid action: {action}")
            return errors

        for param, actions in required_params.items():
            if action in actions and param not in params:
                errors.append(f"{param} is required for {action} action")

        return errors

    async def execute(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute browser operations with reasoning"""
        if params is None:
            params = {}

        # Validate parameters
        errors = self.validate_params(params)
        if errors:
            return {"success": False, "errors": errors}

        # For extension operations, check credentials
        if params.get("action") in ["save_password", "update_autofill"]:
            missing_creds = self.check_credentials()
            if missing_creds:
                return {
                    "success": False,
                    "errors": [f"Missing credentials: {', '.join(missing_creds)}"]
                }

        action = params["action"]
        try:
            # Apply reasoning based on action
            reasoning_result = await self._apply_reasoning(action, params)
            if not reasoning_result["success"]:
                return reasoning_result

            # Execute action with reasoning context
            result = await self._execute_action(action, params, reasoning_result.get("context", {}))
            
            # Update metrics
            self._update_metrics(action, result["success"], params)
            
            # Include metrics in response
            result["metrics"] = self.metrics
            
            return result

        except Exception as e:
            self.metrics["failed_operations"] += 1
            return {
                "success": False,
                "error": str(e),
                "metrics": self.metrics,
                "reasoning": "Encountered system error, operation could not be completed"
            }

    async def _apply_reasoning(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply human-like reasoning to browser operations"""
        try:
            reasoning_context = {
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "goal": self._determine_goal(action, params),
                "priority": self._determine_priority(params),
                "strategy": self._determine_strategy(action, params),
                "considerations": self._determine_considerations(action, params)
            }

            # For bookmark organization, determine optimal folder structure
            if action == "add_bookmark" and "folder" not in params.get("bookmark_data", {}):
                folder_result = await self._determine_bookmark_folder(params)
                if not folder_result["success"]:
                    return folder_result
                if "bookmark_data" not in params:
                    params["bookmark_data"] = {}
                params["bookmark_data"]["folder"] = folder_result["folder"]
                reasoning_context["folder_selection"] = folder_result["reasoning"]

            return {
                "success": True,
                "context": reasoning_context
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"Reasoning error: {str(e)}"],
                "reasoning": "Failed to apply reasoning to the operation"
            }

    def _determine_goal(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive goal of the browser operation"""
        goals = {
            "primary_goal": "",
            "sub_goals": [],
            "success_criteria": [],
            "constraints": [],
            "optimization_targets": []
        }

        if action == "clear_cache":
            goals.update({
                "primary_goal": "Optimize browser performance and free up space",
                "sub_goals": [
                    "Identify non-essential cache",
                    "Preserve important data",
                    "Clear selected items",
                    "Verify performance impact"
                ],
                "success_criteria": [
                    "Space freed up",
                    "Performance improved",
                    "Essential data preserved",
                    "No functionality loss"
                ],
                "constraints": [
                    "Required cache data",
                    "System resources",
                    "User preferences",
                    "Active sessions"
                ],
                "optimization_targets": [
                    "Cache size reduction",
                    "Performance gain",
                    "Data preservation",
                    "Operation speed"
                ]
            })
        elif action == "clear_cookies":
            goals.update({
                "primary_goal": "Enhance privacy and security",
                "sub_goals": [
                    "Identify tracking cookies",
                    "Preserve essential cookies",
                    "Apply cookie policies",
                    "Update preferences"
                ],
                "success_criteria": [
                    "Privacy enhanced",
                    "Security improved",
                    "Preferences maintained",
                    "Policies enforced"
                ],
                "constraints": [
                    "Required cookies",
                    "User sessions",
                    "Site functionality",
                    "Privacy settings"
                ],
                "optimization_targets": [
                    "Privacy protection",
                    "Security enhancement",
                    "Functionality preservation",
                    "Policy compliance"
                ]
            })
        elif action == "add_bookmark":
            goals.update({
                "primary_goal": "Save and organize important web resources",
                "sub_goals": [
                    "Analyze content type",
                    "Determine category",
                    "Select folder",
                    "Update organization"
                ],
                "success_criteria": [
                    "Bookmark saved",
                    "Properly categorized",
                    "Easily accessible",
                    "Organization maintained"
                ],
                "constraints": [
                    "Folder structure",
                    "Naming conventions",
                    "Storage limits",
                    "Sync requirements"
                ],
                "optimization_targets": [
                    "Organization efficiency",
                    "Access speed",
                    "Sync performance",
                    "Storage utilization"
                ]
            })
        elif action == "check_speed":
            goals.update({
                "primary_goal": "Verify internet connection performance",
                "sub_goals": [
                    "Select test servers",
                    "Measure metrics",
                    "Analyze results",
                    "Compare historical data"
                ],
                "success_criteria": [
                    "Accurate measurements",
                    "Complete data",
                    "Reliable results",
                    "Useful insights"
                ],
                "constraints": [
                    "Network conditions",
                    "Server availability",
                    "Time constraints",
                    "Resource usage"
                ],
                "optimization_targets": [
                    "Test accuracy",
                    "Data completeness",
                    "Analysis depth",
                    "Resource efficiency"
                ]
            })
        else:
            goals.update({
                "primary_goal": "Manage browser efficiently and maintain optimal user experience",
                "sub_goals": [
                    "Execute operation",
                    "Maintain performance",
                    "Preserve settings",
                    "Track metrics"
                ],
                "success_criteria": [
                    "Operation completed",
                    "Performance maintained",
                    "Settings preserved",
                    "Metrics updated"
                ],
                "constraints": [
                    "System resources",
                    "User preferences",
                    "Browser settings",
                    "Security policies"
                ],
                "optimization_targets": [
                    "Operation efficiency",
                    "Resource usage",
                    "User experience",
                    "System health"
                ]
            })

        return goals

    def _determine_priority(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive priority of the browser operation"""
        priority_info = {
            "level": "normal",
            "urgency": "low",
            "factors": [],
            "impact": {},
            "handling": {}
        }

        # Check explicit priority
        if "priority" in params:
            priority_info["level"] = params["priority"]
            priority_info["factors"].append("User specified priority")

        # Check memory usage for performance-related operations
        if params.get("action") in ["clear_cache", "clear_cookies"]:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                priority_info["level"] = "high"
                priority_info["urgency"] = "high"
                priority_info["factors"].append(f"High memory usage ({memory_percent}%)")
            elif memory_percent > 80:
                priority_info["urgency"] = "medium"
                priority_info["factors"].append(f"Elevated memory usage ({memory_percent}%)")

        # Check cache size if available
        if "cache_size" in params:
            cache_threshold = self.get_config("settings.cache.max_size_mb", 1024)
            cache_size_mb = params["cache_size"] / (1024 * 1024)
            if cache_size_mb > cache_threshold:
                priority_info["level"] = "high"
                priority_info["factors"].append(f"Cache exceeds threshold ({cache_size_mb:.1f}MB)")

        # Check security factors
        if params.get("action") in ["clear_cookies", "save_password"]:
            if params.get("security_risk"):
                priority_info["level"] = "high"
                priority_info["urgency"] = "high"
                priority_info["factors"].append("Security risk detected")

        # Determine impact
        priority_info["impact"] = {
            "performance_impact": "high" if priority_info["urgency"] == "high" else "normal",
            "security_impact": "significant" if "security_risk" in params else "minimal",
            "user_impact": "immediate" if priority_info["urgency"] == "high" else "standard"
        }

        # Set handling recommendations
        priority_info["handling"] = {
            "execution_order": "immediate" if priority_info["urgency"] == "high" else 
                             "next_in_queue" if priority_info["urgency"] == "medium" else 
                             "normal",
            "backup_first": params.get("action") in ["clear_cache", "clear_cookies"],
            "notify_user": priority_info["level"] == "high",
            "monitoring": priority_info["level"] in ["high", "medium"]
        }

        return priority_info

    def _determine_strategy(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive strategy for handling the browser operation"""
        strategy = {
            "approach": "",
            "steps": [],
            "validations": [],
            "fallback": {},
            "optimization": {},
            "monitoring": {}
        }

        if action == "clear_cache":
            strategy.update({
                "approach": "Clear non-essential cached data while preserving important items",
                "steps": [
                    "Analyze cache content",
                    "Identify essential data",
                    "Clear selected items",
                    "Verify cleanup",
                    "Update metrics"
                ],
                "validations": [
                    "Cache size reduction",
                    "Essential data preserved",
                    "Performance impact",
                    "System stability"
                ],
                "fallback": {
                    "partial_clear": "Clear oldest items first",
                    "preservation_fail": "Backup important data",
                    "performance_impact": "Revert changes"
                },
                "optimization": {
                    "selective_clearing": True,
                    "background_processing": True,
                    "compression": "auto"
                },
                "monitoring": {
                    "track_size": True,
                    "measure_performance": True,
                    "verify_integrity": True
                }
            })
        elif action == "add_bookmark":
            strategy.update({
                "approach": "Organize bookmarks logically for easy access",
                "steps": [
                    "Analyze content",
                    "Determine category",
                    "Select folder",
                    "Create bookmark",
                    "Update organization"
                ],
                "validations": [
                    "URL validity",
                    "Folder structure",
                    "Duplicate check",
                    "Sync status"
                ],
                "fallback": {
                    "category_mismatch": "Use default folder",
                    "sync_fail": "Store locally",
                    "duplicate_found": "Update existing"
                },
                "optimization": {
                    "smart_categorization": True,
                    "auto_organization": True,
                    "sync_efficiency": True
                },
                "monitoring": {
                    "track_usage": True,
                    "check_organization": True,
                    "verify_access": True
                }
            })
        elif action == "check_speed":
            strategy.update({
                "approach": "Run comprehensive speed test with multiple servers",
                "steps": [
                    "Select test servers",
                    "Run download test",
                    "Run upload test",
                    "Measure latency",
                    "Analyze results"
                ],
                "validations": [
                    "Server response",
                    "Connection stability",
                    "Data accuracy",
                    "Result consistency"
                ],
                "fallback": {
                    "server_unavailable": "Try alternate server",
                    "connection_error": "Retry with backoff",
                    "inconsistent_results": "Run additional tests"
                },
                "optimization": {
                    "parallel_testing": True,
                    "adaptive_servers": True,
                    "result_caching": True
                },
                "monitoring": {
                    "track_trends": True,
                    "compare_servers": True,
                    "log_conditions": True
                }
            })

        return strategy

    def _determine_considerations(self, action: str, params: Dict[str, Any]) -> List[str]:
        """Determine important considerations for the browser operation"""
        considerations = []
        
        if action == "clear_cache":
            considerations.extend([
                "Current memory usage",
                "Essential cached data",
                "Performance impact",
                "User preferences"
            ])
        elif action == "add_bookmark":
            considerations.extend([
                "Existing folder structure",
                "Related bookmarks",
                "Frequency of access",
                "Category relevance"
            ])
        elif action == "check_speed":
            considerations.extend([
                "Network conditions",
                "Server selection",
                "Background processes",
                "Time of day"
            ])
            
        return considerations

    async def _determine_bookmark_folder(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal bookmark folder using RAG"""
        try:
            result = self.rag_skill.execute({
                "action": "generate",
                "query": "determine bookmark folder",
                "context": params
            })

            if not result["success"]:
                return result

            return {
                "success": True,
                "folder": result["content"],
                "reasoning": result.get("reasoning", {})
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"Folder determination error: {str(e)}"]
            }

    def _update_metrics(self, action: str, success: bool, params: Dict[str, Any]) -> None:
        """Update comprehensive skill metrics based on action and result"""
        start_time = datetime.now()
        
        if success:
            self.metrics["successful_operations"] += 1
            
            # Update action-specific metrics
            if action == "open_tab":
                self.metrics["tabs_opened"] += 1
                self.metrics["tab_metrics"]["max_tabs_open"] = max(
                    self.metrics["tab_metrics"]["max_tabs_open"],
                    self.metrics["tabs_opened"] - self.metrics["tabs_closed"]
                )
                
                # Update domain statistics if URL provided
                if "url" in params:
                    domain = self._extract_domain(params["url"])
                    self.metrics["tab_metrics"]["most_visited_domains"][domain] = (
                        self.metrics["tab_metrics"]["most_visited_domains"].get(domain, 0) + 1
                    )
                    
                # Update tab category if provided
                if "category" in params:
                    category = params["category"]
                    if category in self.metrics["tab_metrics"]["tab_categories"]:
                        self.metrics["tab_metrics"]["tab_categories"][category] += 1
                    else:
                        self.metrics["tab_metrics"]["tab_categories"]["other"] += 1
                
            elif action == "close_tab":
                self.metrics["tabs_closed"] += 1
                if "tab_lifetime" in params:
                    # Update average tab lifetime
                    current_avg = self.metrics["tab_metrics"]["avg_tab_lifetime"]
                    total_tabs = self.metrics["tabs_closed"]
                    self.metrics["tab_metrics"]["avg_tab_lifetime"] = (
                        (current_avg * (total_tabs - 1) + params["tab_lifetime"]) / total_tabs
                    )
                
            elif action == "clear_cache":
                self.metrics["cache_cleared_count"] += 1
                if "cache_size" in params:
                    self.metrics["performance_metrics"]["cache_size"] = params["cache_size"]
                
            elif action == "clear_cookies":
                self.metrics["cookies_cleared_count"] += 1
                self.metrics["security_metrics"]["cookie_policies_enforced"] += 1
                
            elif action == "login":
                self.metrics["successful_logins"] += 1
                self.metrics["security_metrics"]["logins_protected"] += 1
                if params.get("connection_secure", False):
                    self.metrics["security_metrics"]["secure_connections"] += 1
                else:
                    self.metrics["security_metrics"]["insecure_connections"] += 1
                
            elif action == "save_password":
                self.metrics["passwords_saved"] += 1
                self.metrics["security_metrics"]["passwords_encrypted"] += 1
                
            elif action == "add_bookmark":
                self.metrics["bookmarks_added"] += 1
                if "bookmark_data" in params:
                    folder = params["bookmark_data"].get("folder", "other")
                    self.metrics["bookmark_metrics"]["folder_distribution"][folder] = (
                        self.metrics["bookmark_metrics"]["folder_distribution"].get(folder, 0) + 1
                    )
                    
                    if "url" in params:
                        domain = self._extract_domain(params["url"])
                        self.metrics["bookmark_metrics"]["most_bookmarked_domains"][domain] = (
                            self.metrics["bookmark_metrics"]["most_bookmarked_domains"].get(domain, 0) + 1
                        )
                        
                    if "category" in params["bookmark_data"]:
                        category = params["bookmark_data"]["category"]
                        if category in self.metrics["bookmark_metrics"]["bookmark_categories"]:
                            self.metrics["bookmark_metrics"]["bookmark_categories"][category] += 1
                        else:
                            self.metrics["bookmark_metrics"]["bookmark_categories"]["other"] += 1
                
            elif action == "remove_bookmark":
                self.metrics["bookmarks_removed"] += 1
                if params.get("unused", False):
                    self.metrics["bookmark_metrics"]["unused_bookmarks"] += 1
                
            elif action == "check_speed":
                self.metrics["speed_tests_run"] += 1
                
                if "speed_data" in params:
                    speed_data = params["speed_data"]
                    
                    # Update speed metrics
                    if "download" in speed_data:
                        self.metrics["speed_test_metrics"]["peak_download"] = max(
                            self.metrics["speed_test_metrics"]["peak_download"],
                            speed_data["download"]
                        )
                        
                    if "upload" in speed_data:
                        self.metrics["speed_test_metrics"]["peak_upload"] = max(
                            self.metrics["speed_test_metrics"]["peak_upload"],
                            speed_data["upload"]
                        )
                        
                    if "ping" in speed_data:
                        self.metrics["speed_test_metrics"]["lowest_ping"] = min(
                            self.metrics["speed_test_metrics"]["lowest_ping"],
                            speed_data["ping"]
                        )
                    
                    # Update server statistics
                    if "server" in speed_data:
                        server = speed_data["server"]
                        self.metrics["speed_test_metrics"]["tests_by_server"][server] = (
                            self.metrics["speed_test_metrics"]["tests_by_server"].get(server, 0) + 1
                        )
                    
                    # Update time of day performance
                    hour = datetime.now().hour
                    if 6 <= hour < 12:
                        period = "morning"
                    elif 12 <= hour < 17:
                        period = "afternoon"
                    elif 17 <= hour < 22:
                        period = "evening"
                    else:
                        period = "night"
                    
                    current_avg = self.metrics["speed_test_metrics"]["time_of_day_performance"][period]
                    tests_in_period = self.metrics["speed_test_metrics"]["tests_by_server"].get(period, 0)
                    self.metrics["speed_test_metrics"]["time_of_day_performance"][period] = (
                        (current_avg * tests_in_period + speed_data.get("download", 0)) / (tests_in_period + 1)
                    )
                    
            elif action == "skip_ad":
                self.metrics["ads_skipped"] += 1
            elif action == "copy_link":
                self.metrics["links_copied"] += 1
            elif action == "update_autofill":
                self.metrics["autofills_updated"] += 1

            # Update performance metrics
            if "performance_data" in params:
                perf_data = params["performance_data"]
                self.metrics["performance_metrics"].update({
                    "avg_page_load_time": (
                        self.metrics["performance_metrics"]["avg_page_load_time"] * 0.8 +
                        perf_data.get("load_time", 0) * 0.2
                    ),
                    "memory_usage": perf_data.get("memory", 0),
                    "cpu_usage": perf_data.get("cpu", 0),
                    "bandwidth_usage": perf_data.get("bandwidth", 0),
                    "cache_hit_rate": perf_data.get("cache_hits", 0)
                })

        else:
            self.metrics["failed_operations"] += 1

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"

    async def _execute_action(self, action: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute browser action with context"""
        try:
            if action == "open_tab":
                return await self._open_tab(params, context)
            elif action == "close_tab":
                return await self._close_tab(params, context)
            elif action == "clear_cache":
                return await self._clear_cache(context)
            elif action == "clear_cookies":
                return await self._clear_cookies(context)
            elif action == "login":
                return await self._login(params, context)
            elif action == "logout":
                return await self._logout(context)
            elif action == "save_password":
                return await self._save_password(params, context)
            elif action == "update_autofill":
                return await self._update_autofill(params, context)
            elif action == "add_bookmark":
                return await self._add_bookmark(params, context)
            elif action == "remove_bookmark":
                return await self._remove_bookmark(params, context)
            elif action == "check_speed":
                return await self._check_speed(context)
            elif action == "skip_ad":
                return await self._skip_ad(context)
            elif action == "copy_link":
                return await self._copy_link(context)
            else:
                return {
                    "success": False,
                    "errors": [f"Unknown action: {action}"]
                }
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "context": context
            }

    async def _open_tab(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Open a new browser tab with context"""
        try:
            # Implementation would open new tab with specified URL
            return {
                "success": True,
                "message": "Tab opened successfully",
                "context": context
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "context": context
            }

    async def _check_speed(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check internet speed with context"""
        try:
            # Implementation would perform speed test
            # For now, returning mock success
            speed_info = {
                "download": 50.5,  # Mbps
                "upload": 10.2,    # Mbps
                "ping": 20,        # ms
                "server": self.get_config("settings.speed_test.test_server", "speedtest.net")
            }
            
            return {
                "success": True,
                "message": "Speed check completed",
                "speed_info": speed_info,
                "context": context
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "context": context
            }

    def cleanup(self):
        """Clean up resources"""
        # Implementation would clean up any resources
        pass
