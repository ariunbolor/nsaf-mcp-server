from typing import Dict, Any, List, Optional
from datetime import datetime
import subprocess
import os
import platform
import psutil
from .base_skill import BaseSkill
from .rag_skill import RAGSkill

class SystemMaintenanceSkill(BaseSkill):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="system_maintenance",
            description="A skill for managing system maintenance tasks including security, updates, and system optimization",
            config=config
        )
        self.required_credentials = [
            "admin_password"  # For system-level operations
        ]
        self.rag_skill = RAGSkill()
        self.metrics = {
            "virus_scans_completed": 0,
            "threats_detected": 0,
            "threats_resolved": 0,
            "updates_installed": 0,
            "system_restarts": 0,
            "performance_checks": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "space_cleaned": 0,  # in bytes
            "avg_cpu_usage": 0,  # percentage
            "avg_memory_usage": 0,  # percentage
            "battery_health": 100,  # percentage
            "security_metrics": {
                "scan_coverage": 0.0,  # percentage
                "threat_types": {},    # type: count
                "detection_sources": {},  # source: count
                "quarantined_items": 0,
                "false_positives": 0,
                "scan_durations": [],  # list of durations in seconds
                "high_risk_areas": {}  # area: risk_score
            },
            "update_metrics": {
                "critical_updates": 0,
                "security_updates": 0,
                "feature_updates": 0,
                "failed_updates": 0,
                "rollbacks": 0,
                "update_sizes": {},  # update_type: size in bytes
                "update_durations": {}  # update_type: duration in seconds
            },
            "performance_metrics": {
                "cpu_metrics": {
                    "peak_usage": 0.0,
                    "usage_trend": [],  # list of {timestamp, usage}
                    "core_distribution": {},  # core: usage
                    "process_impact": {}  # process: cpu_usage
                },
                "memory_metrics": {
                    "peak_usage": 0.0,
                    "usage_trend": [],
                    "swap_usage": 0.0,
                    "page_faults": 0,
                    "largest_consumers": {}  # process: memory_usage
                },
                "disk_metrics": {
                    "read_speed": 0.0,  # MB/s
                    "write_speed": 0.0,  # MB/s
                    "io_operations": 0,
                    "fragmentation": 0.0,  # percentage
                    "smart_status": {}  # attribute: value
                },
                "network_metrics": {
                    "bandwidth_usage": 0.0,  # MB/s
                    "latency": 0.0,  # ms
                    "packet_loss": 0.0,  # percentage
                    "active_connections": 0
                }
            },
            "system_health": {
                "overall_score": 100,  # 0-100
                "component_scores": {
                    "cpu_health": 100,
                    "memory_health": 100,
                    "disk_health": 100,
                    "battery_health": 100,
                    "security_health": 100
                },
                "warnings": [],
                "critical_issues": []
            },
            "maintenance_metrics": {
                "scheduled_tasks": 0,
                "automated_fixes": 0,
                "manual_interventions": 0,
                "preventive_actions": 0,
                "task_durations": {},  # task_type: avg_duration
                "success_rates": {},   # task_type: success_rate
                "resource_usage": {}   # task_type: resource_impact
            },
            "optimization_metrics": {
                "space_reclaimed": 0,  # in bytes
                "startup_time": 0.0,   # in seconds
                "shutdown_time": 0.0,  # in seconds
                "boot_optimizations": 0,
                "performance_gains": {
                    "cpu": 0.0,  # percentage improvement
                    "memory": 0.0,
                    "disk": 0.0,
                    "network": 0.0
                }
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters for system maintenance operations"""
        errors = []
        required_params = {
            "action": ["run_virus_scan", "update_antivirus", "check_system_updates",
                      "restart_system", "clean_desktop", "change_wallpaper",
                      "adjust_brightness", "manage_peripherals", "check_performance",
                      "check_battery"],
            "scan_type": ["run_virus_scan"],
            "brightness_level": ["adjust_brightness"],
            "wallpaper_path": ["change_wallpaper"],
            "peripheral_action": ["manage_peripherals"],
            "peripheral_id": ["manage_peripherals"]
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
        """Execute system maintenance operations with reasoning"""
        if params is None:
            params = {}

        # Validate parameters
        errors = self.validate_params(params)
        if errors:
            return {"success": False, "errors": errors}

        # For system operations, check credentials
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
        """Apply human-like reasoning to system maintenance operations"""
        try:
            reasoning_context = {
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "goal": self._determine_goal(action, params),
                "priority": self._determine_priority(params),
                "strategy": self._determine_strategy(action, params),
                "considerations": self._determine_considerations(action, params)
            }

            # For performance optimization, get recommendations
            if action in ["check_performance", "clean_desktop"]:
                opt_result = await self._get_optimization_recommendations(params)
                if not opt_result["success"]:
                    return opt_result
                reasoning_context["optimization"] = opt_result["recommendations"]
                reasoning_context["impact"] = opt_result["impact"]

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
        """Determine the comprehensive goal of the system maintenance operation"""
        goals = {
            "primary_goal": "",
            "sub_goals": [],
            "success_criteria": [],
            "constraints": [],
            "optimization_targets": []
        }

        if action == "run_virus_scan":
            goals.update({
                "primary_goal": "Ensure system security and detect potential threats",
                "sub_goals": [
                    "Scan critical areas",
                    "Identify threats",
                    "Analyze risks",
                    "Take protective action"
                ],
                "success_criteria": [
                    "Complete scan coverage",
                    "Threats identified",
                    "Risks mitigated",
                    "System protected"
                ],
                "constraints": [
                    "System resources",
                    "Scan duration",
                    "Critical processes",
                    "User activity"
                ],
                "optimization_targets": [
                    "Scan efficiency",
                    "Detection accuracy",
                    "Resource usage",
                    "User impact"
                ]
            })
        elif action == "check_performance":
            goals.update({
                "primary_goal": "Optimize system performance and resource usage",
                "sub_goals": [
                    "Monitor resources",
                    "Identify bottlenecks",
                    "Analyze patterns",
                    "Optimize usage"
                ],
                "success_criteria": [
                    "Resource efficiency",
                    "System responsiveness",
                    "Process optimization",
                    "User experience"
                ],
                "constraints": [
                    "Available resources",
                    "System load",
                    "Application needs",
                    "Power consumption"
                ],
                "optimization_targets": [
                    "CPU efficiency",
                    "Memory usage",
                    "Disk performance",
                    "Network throughput"
                ]
            })
        elif action == "clean_desktop":
            goals.update({
                "primary_goal": "Improve organization and system efficiency",
                "sub_goals": [
                    "Analyze content",
                    "Categorize items",
                    "Optimize layout",
                    "Remove clutter"
                ],
                "success_criteria": [
                    "Organized content",
                    "Efficient access",
                    "Space optimized",
                    "User friendly"
                ],
                "constraints": [
                    "User preferences",
                    "File importance",
                    "Access patterns",
                    "Space limits"
                ],
                "optimization_targets": [
                    "Organization clarity",
                    "Access speed",
                    "Space usage",
                    "Visual appeal"
                ]
            })
        elif action == "check_battery":
            goals.update({
                "primary_goal": "Monitor and optimize power consumption",
                "sub_goals": [
                    "Check health",
                    "Analyze usage",
                    "Optimize settings",
                    "Extend life"
                ],
                "success_criteria": [
                    "Battery health",
                    "Power efficiency",
                    "Usage optimization",
                    "Longevity improved"
                ],
                "constraints": [
                    "Battery capacity",
                    "Usage patterns",
                    "System needs",
                    "Temperature"
                ],
                "optimization_targets": [
                    "Power efficiency",
                    "Heat management",
                    "Charge cycles",
                    "Runtime duration"
                ]
            })
        else:
            goals.update({
                "primary_goal": "Maintain system health and performance",
                "sub_goals": [
                    "Monitor health",
                    "Optimize resources",
                    "Prevent issues",
                    "Improve efficiency"
                ],
                "success_criteria": [
                    "System stability",
                    "Resource efficiency",
                    "Issue prevention",
                    "Performance gain"
                ],
                "constraints": [
                    "System resources",
                    "User needs",
                    "Application requirements",
                    "Environmental factors"
                ],
                "optimization_targets": [
                    "System health",
                    "Resource usage",
                    "User experience",
                    "Longevity"
                ]
            })

        return goals

    def _determine_priority(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive priority of the system maintenance operation"""
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

        # Check system state for priority determination
        if params.get("action") in ["run_virus_scan", "check_performance"]:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage("/").percent
            
            if cpu_usage > 90:
                priority_info["level"] = "high"
                priority_info["urgency"] = "high"
                priority_info["factors"].append(f"Critical CPU usage ({cpu_usage}%)")
            elif cpu_usage > 80:
                priority_info["urgency"] = "medium"
                priority_info["factors"].append(f"High CPU usage ({cpu_usage}%)")
                
            if memory_usage > 90:
                priority_info["level"] = "high"
                priority_info["urgency"] = "high"
                priority_info["factors"].append(f"Critical memory usage ({memory_usage}%)")
            elif memory_usage > 80:
                priority_info["urgency"] = "medium"
                priority_info["factors"].append(f"High memory usage ({memory_usage}%)")
                
            if disk_usage > 90:
                priority_info["level"] = "high"
                priority_info["urgency"] = "high"
                priority_info["factors"].append(f"Critical disk usage ({disk_usage}%)")
            elif disk_usage > 80:
                priority_info["urgency"] = "medium"
                priority_info["factors"].append(f"High disk usage ({disk_usage}%)")

        # Check security factors
        if params.get("action") == "run_virus_scan":
            if params.get("threat_detected", False):
                priority_info["level"] = "high"
                priority_info["urgency"] = "high"
                priority_info["factors"].append("Active threat detected")
            if params.get("scan_overdue", False):
                priority_info["urgency"] = "medium"
                priority_info["factors"].append("Scan overdue")

        # Check battery factors
        if params.get("action") == "check_battery":
            if params.get("battery_level", 100) < 20:
                priority_info["level"] = "high"
                priority_info["factors"].append("Critical battery level")
            if params.get("battery_health", 100) < 50:
                priority_info["level"] = "high"
                priority_info["factors"].append("Poor battery health")

        # Determine impact
        priority_info["impact"] = {
            "system_impact": "high" if priority_info["level"] == "high" else "normal",
            "performance_impact": "significant" if priority_info["urgency"] == "high" else "minimal",
            "user_impact": "immediate" if priority_info["urgency"] == "high" else "standard",
            "security_impact": "critical" if "threat_detected" in priority_info["factors"] else "normal"
        }

        # Set handling recommendations
        priority_info["handling"] = {
            "execution_order": "immediate" if priority_info["urgency"] == "high" else 
                             "next_in_queue" if priority_info["urgency"] == "medium" else 
                             "normal",
            "resource_allocation": "high" if priority_info["level"] == "high" else "normal",
            "monitoring_level": "intensive" if priority_info["level"] == "high" else "standard",
            "notification_required": priority_info["level"] in ["high", "medium"],
            "backup_recommended": priority_info["level"] == "high"
        }

        return priority_info

    def _determine_strategy(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive strategy for handling the system maintenance operation"""
        strategy = {
            "approach": "",
            "steps": [],
            "validations": [],
            "fallback": {},
            "optimization": {},
            "monitoring": {}
        }

        if action == "run_virus_scan":
            strategy.update({
                "approach": "Scan critical areas first, then perform thorough scan",
                "steps": [
                    "Check critical areas",
                    "Scan system files",
                    "Analyze threats",
                    "Take action",
                    "Verify system"
                ],
                "validations": [
                    "Scan completion",
                    "Threat detection",
                    "System integrity",
                    "Resource impact"
                ],
                "fallback": {
                    "scan_interrupted": "Resume from checkpoint",
                    "resource_constraint": "Reduce scope",
                    "detection_unclear": "Deep scan",
                    "system_busy": "Defer non-critical"
                },
                "optimization": {
                    "parallel_scanning": True,
                    "resource_management": "adaptive",
                    "priority_areas": True,
                    "incremental_scan": True
                },
                "monitoring": {
                    "scan_progress": True,
                    "system_health": True,
                    "resource_usage": True,
                    "threat_detection": True
                }
            })
        elif action == "check_performance":
            strategy.update({
                "approach": "Analyze resource usage and identify bottlenecks",
                "steps": [
                    "Monitor resources",
                    "Analyze patterns",
                    "Identify issues",
                    "Optimize usage",
                    "Verify improvements"
                ],
                "validations": [
                    "Resource efficiency",
                    "System responsiveness",
                    "Application performance",
                    "User experience"
                ],
                "fallback": {
                    "high_usage": "Throttle processes",
                    "memory_pressure": "Free cache",
                    "disk_bottleneck": "Optimize I/O",
                    "network_issues": "Reduce bandwidth"
                },
                "optimization": {
                    "process_priority": True,
                    "memory_management": True,
                    "io_scheduling": True,
                    "power_efficiency": True
                },
                "monitoring": {
                    "resource_trends": True,
                    "bottleneck_detection": True,
                    "performance_metrics": True,
                    "system_health": True
                }
            })
        elif action == "clean_desktop":
            strategy.update({
                "approach": "Organize files by type and access frequency",
                "steps": [
                    "Analyze content",
                    "Sort items",
                    "Create structure",
                    "Move files",
                    "Verify organization"
                ],
                "validations": [
                    "File accessibility",
                    "Organization logic",
                    "Space efficiency",
                    "User workflow"
                ],
                "fallback": {
                    "space_issue": "Archive old files",
                    "access_error": "Skip problem files",
                    "type_unknown": "Use misc category",
                    "user_busy": "Background processing"
                },
                "optimization": {
                    "batch_operations": True,
                    "smart_categorization": True,
                    "access_patterns": True,
                    "space_efficiency": True
                },
                "monitoring": {
                    "organization_progress": True,
                    "space_usage": True,
                    "file_access": True,
                    "user_interaction": True
                }
            })

        return strategy

    def _determine_considerations(self, action: str, params: Dict[str, Any]) -> List[str]:
        """Determine important considerations for the system maintenance operation"""
        considerations = []
        
        if action == "run_virus_scan":
            considerations.extend([
                "System resource usage",
                "Critical system areas",
                "Recent suspicious activity",
                "Scan history"
            ])
        elif action == "check_performance":
            considerations.extend([
                "CPU utilization",
                "Memory usage",
                "Disk space",
                "Running processes"
            ])
        elif action == "clean_desktop":
            considerations.extend([
                "File types",
                "Access patterns",
                "Available space",
                "User preferences"
            ])
            
        return considerations

    async def _get_optimization_recommendations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get system optimization recommendations using RAG"""
        try:
            result = self.rag_skill.execute({
                "action": "generate",
                "query": "system optimization recommendations",
                "context": params
            })

            if not result["success"]:
                return result

            return {
                "success": True,
                "recommendations": result["content"],
                "impact": result.get("impact", {}),
                "reasoning": result.get("reasoning", {})
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"Optimization recommendation error: {str(e)}"]
            }

    def _update_metrics(self, action: str, success: bool, params: Dict[str, Any]) -> None:
        """Update comprehensive skill metrics based on action and result"""
        start_time = datetime.now()
        
        if success:
            self.metrics["successful_operations"] += 1
            
            # Update action-specific metrics
            if action == "run_virus_scan":
                self.metrics["virus_scans_completed"] += 1
                
                if "scan_data" in params:
                    scan_data = params["scan_data"]
                    # Update security metrics
                    self.metrics["security_metrics"].update({
                        "scan_coverage": scan_data.get("coverage", 0.0),
                        "quarantined_items": self.metrics["security_metrics"]["quarantined_items"] + 
                                         scan_data.get("quarantined", 0)
                    })
                    
                    # Update threat types
                    if "threat_types" in scan_data:
                        for threat_type, count in scan_data["threat_types"].items():
                            self.metrics["security_metrics"]["threat_types"][threat_type] = (
                                self.metrics["security_metrics"]["threat_types"].get(threat_type, 0) + count
                            )
                    
                    # Update scan duration
                    if "duration" in scan_data:
                        self.metrics["security_metrics"]["scan_durations"].append(scan_data["duration"])
                
                if "threats_found" in params:
                    self.metrics["threats_detected"] += params["threats_found"]
                if "threats_resolved" in params:
                    self.metrics["threats_resolved"] += params["threats_resolved"]
                
            elif action == "check_system_updates":
                if "update_data" in params:
                    update_data = params["update_data"]
                    self.metrics["update_metrics"].update({
                        "critical_updates": self.metrics["update_metrics"]["critical_updates"] + 
                                        update_data.get("critical", 0),
                        "security_updates": self.metrics["update_metrics"]["security_updates"] + 
                                        update_data.get("security", 0),
                        "feature_updates": self.metrics["update_metrics"]["feature_updates"] + 
                                       update_data.get("feature", 0)
                    })
                    
                    # Update size and duration metrics
                    if "sizes" in update_data:
                        for update_type, size in update_data["sizes"].items():
                            self.metrics["update_metrics"]["update_sizes"][update_type] = size
                    if "durations" in update_data:
                        for update_type, duration in update_data["durations"].items():
                            self.metrics["update_metrics"]["update_durations"][update_type] = duration
                
            elif action == "restart_system":
                self.metrics["system_restarts"] += 1
                
            elif action == "check_performance":
                self.metrics["performance_checks"] += 1
                
                # Get current performance data
                cpu_usage = psutil.cpu_percent(percpu=True)
                memory = psutil.virtual_memory()
                disk = psutil.disk_io_counters()
                network = psutil.net_io_counters()
                
                # Update CPU metrics
                self.metrics["performance_metrics"]["cpu_metrics"].update({
                    "peak_usage": max(
                        self.metrics["performance_metrics"]["cpu_metrics"]["peak_usage"],
                        max(cpu_usage)
                    ),
                    "core_distribution": dict(enumerate(cpu_usage))
                })
                self.metrics["performance_metrics"]["cpu_metrics"]["usage_trend"].append({
                    "timestamp": datetime.now().isoformat(),
                    "usage": sum(cpu_usage) / len(cpu_usage)
                })
                
                # Update memory metrics
                self.metrics["performance_metrics"]["memory_metrics"].update({
                    "peak_usage": max(
                        self.metrics["performance_metrics"]["memory_metrics"]["peak_usage"],
                        memory.percent
                    ),
                    "swap_usage": memory.swap_percent if hasattr(memory, 'swap_percent') else 0.0,
                    "page_faults": memory.page_faults if hasattr(memory, 'page_faults') else 0
                })
                self.metrics["performance_metrics"]["memory_metrics"]["usage_trend"].append({
                    "timestamp": datetime.now().isoformat(),
                    "usage": memory.percent
                })
                
                # Update disk metrics
                self.metrics["performance_metrics"]["disk_metrics"].update({
                    "read_speed": disk.read_bytes / 1024 / 1024,  # Convert to MB/s
                    "write_speed": disk.write_bytes / 1024 / 1024,
                    "io_operations": disk.read_count + disk.write_count
                })
                
                # Update network metrics
                self.metrics["performance_metrics"]["network_metrics"].update({
                    "bandwidth_usage": (network.bytes_sent + network.bytes_recv) / 1024 / 1024,
                    "packet_loss": network.dropin + network.dropout,
                    "active_connections": len(psutil.net_connections())
                })
                
                # Update average usage metrics
                self.metrics["avg_cpu_usage"] = (
                    self.metrics["avg_cpu_usage"] * (self.metrics["performance_checks"] - 1) + 
                    sum(cpu_usage) / len(cpu_usage)
                ) / self.metrics["performance_checks"]
                
                self.metrics["avg_memory_usage"] = (
                    self.metrics["avg_memory_usage"] * (self.metrics["performance_checks"] - 1) + 
                    memory.percent
                ) / self.metrics["performance_checks"]
                
            elif action == "clean_desktop":
                if "cleanup_data" in params:
                    cleanup = params["cleanup_data"]
                    self.metrics["space_cleaned"] += cleanup.get("space_freed", 0)
                    self.metrics["optimization_metrics"]["space_reclaimed"] += cleanup.get("space_freed", 0)
                    
            elif action == "check_battery":
                if "battery_data" in params:
                    battery = params["battery_data"]
                    self.metrics["battery_health"] = battery.get("health", 100)
                    self.metrics["system_health"]["component_scores"]["battery_health"] = battery.get("health", 100)

            # Update system health metrics
            self._update_system_health(params)
            
            # Update maintenance metrics
            self.metrics["maintenance_metrics"]["scheduled_tasks"] += 1
            if "automation_data" in params:
                auto = params["automation_data"]
                self.metrics["maintenance_metrics"].update({
                    "automated_fixes": self.metrics["maintenance_metrics"]["automated_fixes"] + 
                                   auto.get("fixes", 0),
                    "preventive_actions": self.metrics["maintenance_metrics"]["preventive_actions"] + 
                                      auto.get("preventive", 0)
                })
            
            # Update task duration metrics
            operation_time = (datetime.now() - start_time).total_seconds()
            if action in self.metrics["maintenance_metrics"]["task_durations"]:
                current_avg = self.metrics["maintenance_metrics"]["task_durations"][action]
                task_count = self.metrics["successful_operations"]
                self.metrics["maintenance_metrics"]["task_durations"][action] = (
                    (current_avg * (task_count - 1) + operation_time) / task_count
                )
            else:
                self.metrics["maintenance_metrics"]["task_durations"][action] = operation_time
            
            # Update success rates
            total_ops = self.metrics["successful_operations"] + self.metrics["failed_operations"]
            for task_type in self.metrics["maintenance_metrics"]["success_rates"]:
                self.metrics["maintenance_metrics"]["success_rates"][task_type] = (
                    self.metrics["successful_operations"] / total_ops * 100
                )

        else:
            self.metrics["failed_operations"] += 1
            
            # Update error metrics for specific actions
            if action == "check_system_updates":
                self.metrics["update_metrics"]["failed_updates"] += 1
            
            # Update success rates on failure
            total_ops = self.metrics["successful_operations"] + self.metrics["failed_operations"]
            for task_type in self.metrics["maintenance_metrics"]["success_rates"]:
                self.metrics["maintenance_metrics"]["success_rates"][task_type] = (
                    self.metrics["successful_operations"] / total_ops * 100
                )

    def _update_system_health(self, params: Dict[str, Any]) -> None:
        """Update system health scores based on metrics and thresholds"""
        # Update CPU health
        cpu_score = 100 - (self.metrics["performance_metrics"]["cpu_metrics"]["peak_usage"] or 0)
        self.metrics["system_health"]["component_scores"]["cpu_health"] = max(0, cpu_score)
        
        # Update memory health
        memory_score = 100 - (self.metrics["performance_metrics"]["memory_metrics"]["peak_usage"] or 0)
        self.metrics["system_health"]["component_scores"]["memory_health"] = max(0, memory_score)
        
        # Update disk health
        disk_usage = psutil.disk_usage("/").percent
        disk_score = 100 - disk_usage
        self.metrics["system_health"]["component_scores"]["disk_health"] = max(0, disk_score)
        
        # Update security health
        if self.metrics["threats_detected"] > 0:
            security_score = 100 * (
                self.metrics["threats_resolved"] / self.metrics["threats_detected"]
            )
        else:
            security_score = 100
        self.metrics["system_health"]["component_scores"]["security_health"] = max(0, security_score)
        
        # Calculate overall health score (weighted average)
        weights = {
            "cpu_health": 0.2,
            "memory_health": 0.2,
            "disk_health": 0.2,
            "battery_health": 0.2,
            "security_health": 0.2
        }
        
        overall_score = sum(
            score * weights[component]
            for component, score in self.metrics["system_health"]["component_scores"].items()
        )
        
        self.metrics["system_health"]["overall_score"] = max(0, min(100, overall_score))
        
        # Update warnings and critical issues
        self.metrics["system_health"]["warnings"] = []
        self.metrics["system_health"]["critical_issues"] = []
        
        # Check components for issues
        for component, score in self.metrics["system_health"]["component_scores"].items():
            if score < 50:
                self.metrics["system_health"]["critical_issues"].append(
                    f"Critical {component.replace('_', ' ')} issue detected"
                )
            elif score < 70:
                self.metrics["system_health"]["warnings"].append(
                    f"Warning: {component.replace('_', ' ')} needs attention"
                )

    async def _execute_action(self, action: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute system maintenance action with context"""
        try:
            if action == "run_virus_scan":
                return await self._run_virus_scan(params, context)
            elif action == "update_antivirus":
                return await self._update_antivirus(context)
            elif action == "check_system_updates":
                return await self._check_system_updates(context)
            elif action == "restart_system":
                return await self._restart_system(context)
            elif action == "clean_desktop":
                return await self._clean_desktop(context)
            elif action == "change_wallpaper":
                return await self._change_wallpaper(params, context)
            elif action == "adjust_brightness":
                return await self._adjust_brightness(params, context)
            elif action == "manage_peripherals":
                return await self._manage_peripherals(params, context)
            elif action == "check_performance":
                return await self._check_performance(context)
            elif action == "check_battery":
                return await self._check_battery(context)
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

    async def _check_performance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check system performance with context"""
        try:
            # Get system performance metrics
            performance_info = {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage("/").percent,
                "running_processes": len(psutil.process_iter()),
                "load_average": os.getloadavg() if platform.system() != "Windows" else None
            }
            
            # Check against thresholds
            thresholds = self.get_config("settings.performance.alert_thresholds", {})
            alerts = []
            
            if performance_info["cpu_usage"] > thresholds.get("cpu_usage", 90):
                alerts.append("High CPU usage")
            
            if performance_info["memory_usage"] > thresholds.get("memory_usage", 90):
                alerts.append("High memory usage")
            
            if performance_info["disk_usage"] > thresholds.get("disk_usage", 90):
                alerts.append("High disk usage")
            
            return {
                "success": True,
                "message": "Performance check completed",
                "performance_info": performance_info,
                "alerts": alerts,
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
