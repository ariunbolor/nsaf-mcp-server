from typing import Dict, Any, List, Optional
from datetime import datetime
import subprocess
import json
import os
import psutil
from .base_skill import BaseSkill
from .rag_skill import RAGSkill

class SoftwareManagementSkill(BaseSkill):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="software_management",
            description="A skill for managing software operations including updates, installations, and configurations",
            config=config
        )
        self.required_credentials = [
            "admin_password"  # For operations requiring elevated privileges
        ]
        self.rag_skill = RAGSkill()
        self.metrics = {
            "installations_completed": 0,
            "updates_completed": 0,
            "uninstallations_completed": 0,
            "extensions_managed": 0,
            "settings_changed": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "disk_space_saved": 0,  # in bytes
            "total_installation_time": 0,  # in seconds
            "compatibility_checks_run": 0,
            "package_metrics": {
                "packages_installed": {},  # package_name: count
                "packages_updated": {},    # package_name: count
                "packages_removed": {},    # package_name: count
                "preferred_sources": {},   # source: count
                "installation_sizes": {},  # package_name: size in bytes
                "update_sizes": {}        # package_name: size in bytes
            },
            "performance_metrics": {
                "avg_install_time": 0.0,  # in seconds
                "avg_update_time": 0.0,   # in seconds
                "peak_cpu_usage": 0.0,    # percentage
                "peak_memory_usage": 0.0,  # in MB
                "disk_io": 0.0,           # in MB/s
                "network_io": 0.0         # in MB/s
            },
            "dependency_metrics": {
                "dependencies_installed": 0,
                "dependencies_updated": 0,
                "dependency_conflicts": 0,
                "shared_dependencies": {},  # dependency: [packages]
                "orphaned_dependencies": 0
            },
            "system_impact": {
                "bootup_time_change": 0.0,  # in seconds
                "memory_footprint": 0.0,    # in MB
                "disk_usage_trend": [],     # list of {timestamp, usage}
                "cpu_load_impact": 0.0,     # percentage
                "network_impact": 0.0       # in MB/s
            },
            "error_metrics": {
                "installation_errors": {},  # error_type: count
                "update_errors": {},       # error_type: count
                "compatibility_errors": {},  # error_type: count
                "permission_errors": 0,
                "network_errors": 0,
                "disk_space_errors": 0
            },
            "security_metrics": {
                "verified_packages": 0,
                "unverified_packages": 0,
                "security_updates": 0,
                "vulnerability_patches": 0,
                "signature_validations": 0
            },
            "backup_metrics": {
                "backups_created": 0,
                "backups_restored": 0,
                "backup_size_total": 0,  # in bytes
                "successful_rollbacks": 0,
                "failed_rollbacks": 0
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters for software operations"""
        errors = []
        required_params = {
            "action": ["update", "restart", "install", "uninstall", "check_compatibility",
                      "setup_auto_update", "manage_extension", "change_settings",
                      "accept_terms", "subscribe_notifications", "enter_product_key"],
            "software_name": ["update", "restart", "install", "uninstall", 
                            "check_compatibility", "setup_auto_update", 
                            "change_settings", "enter_product_key"],
            "extension_data": ["manage_extension"],
            "settings_data": ["change_settings"],
            "product_key": ["enter_product_key"]
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
        """Execute software management operations with reasoning"""
        if params is None:
            params = {}

        # Validate parameters
        errors = self.validate_params(params)
        if errors:
            return {"success": False, "errors": errors}

        # For privileged operations, check credentials
        privileged_actions = ["install", "uninstall", "update"]
        if params.get("action") in privileged_actions:
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
        """Apply human-like reasoning to software operations"""
        try:
            reasoning_context = {
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "goal": self._determine_goal(action, params),
                "priority": self._determine_priority(params),
                "strategy": self._determine_strategy(action, params),
                "considerations": self._determine_considerations(action, params)
            }

            # For installation/update, check compatibility first
            if action in ["install", "update"]:
                compat_result = await self._check_compatibility_with_rag(params)
                if not compat_result["success"]:
                    return compat_result
                reasoning_context["compatibility"] = compat_result["compatibility"]
                reasoning_context["requirements"] = compat_result["requirements"]

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
        """Determine the comprehensive goal of the software operation"""
        goals = {
            "primary_goal": "",
            "sub_goals": [],
            "success_criteria": [],
            "constraints": [],
            "optimization_targets": []
        }

        if action == "install":
            goals.update({
                "primary_goal": "Add new software functionality while ensuring system stability",
                "sub_goals": [
                    "Verify system requirements",
                    "Resolve dependencies",
                    "Prepare installation",
                    "Configure software"
                ],
                "success_criteria": [
                    "Software installed correctly",
                    "Dependencies satisfied",
                    "System stable",
                    "Performance maintained"
                ],
                "constraints": [
                    "Available disk space",
                    "System resources",
                    "Network bandwidth",
                    "User permissions"
                ],
                "optimization_targets": [
                    "Installation speed",
                    "Resource usage",
                    "Disk efficiency",
                    "Network efficiency"
                ]
            })
        elif action == "update":
            goals.update({
                "primary_goal": "Improve software while maintaining existing configurations",
                "sub_goals": [
                    "Backup current state",
                    "Download updates",
                    "Apply changes",
                    "Verify functionality"
                ],
                "success_criteria": [
                    "Update completed",
                    "Settings preserved",
                    "Functionality verified",
                    "Performance improved"
                ],
                "constraints": [
                    "Existing configuration",
                    "System compatibility",
                    "Service continuity",
                    "Resource limits"
                ],
                "optimization_targets": [
                    "Update speed",
                    "Downtime minimization",
                    "Configuration preservation",
                    "Resource efficiency"
                ]
            })
        elif action == "uninstall":
            goals.update({
                "primary_goal": "Remove software while preserving system integrity",
                "sub_goals": [
                    "Backup data",
                    "Remove components",
                    "Clean dependencies",
                    "Restore system"
                ],
                "success_criteria": [
                    "Software removed",
                    "System cleaned",
                    "Resources freed",
                    "No side effects"
                ],
                "constraints": [
                    "Shared dependencies",
                    "System stability",
                    "Data preservation",
                    "Service impact"
                ],
                "optimization_targets": [
                    "Cleanup thoroughness",
                    "System preservation",
                    "Resource recovery",
                    "Impact minimization"
                ]
            })
        elif action == "check_compatibility":
            goals.update({
                "primary_goal": "Ensure software will run efficiently on the system",
                "sub_goals": [
                    "Analyze requirements",
                    "Check resources",
                    "Verify compatibility",
                    "Assess impact"
                ],
                "success_criteria": [
                    "Requirements met",
                    "Resources available",
                    "Compatibility confirmed",
                    "Impact acceptable"
                ],
                "constraints": [
                    "System specifications",
                    "Resource availability",
                    "Platform support",
                    "Performance needs"
                ],
                "optimization_targets": [
                    "Analysis accuracy",
                    "Resource assessment",
                    "Impact prediction",
                    "Risk evaluation"
                ]
            })
        else:
            goals.update({
                "primary_goal": "Manage software effectively while maintaining system performance",
                "sub_goals": [
                    "Execute operation",
                    "Monitor system",
                    "Track changes",
                    "Maintain stability"
                ],
                "success_criteria": [
                    "Operation completed",
                    "System stable",
                    "Changes tracked",
                    "Performance maintained"
                ],
                "constraints": [
                    "System resources",
                    "User requirements",
                    "Service levels",
                    "Security policies"
                ],
                "optimization_targets": [
                    "Operation efficiency",
                    "Resource usage",
                    "System health",
                    "User experience"
                ]
            })

        return goals

    def _determine_priority(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive priority of the software operation"""
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

        # Check system resources for installation/update operations
        if params.get("action") in ["install", "update"]:
            disk_space = psutil.disk_usage("/").percent
            memory_usage = psutil.virtual_memory().percent
            
            if disk_space > 90:
                priority_info["level"] = "high"
                priority_info["urgency"] = "high"
                priority_info["factors"].append(f"Critical disk space ({disk_space}%)")
            elif disk_space > 80:
                priority_info["urgency"] = "medium"
                priority_info["factors"].append(f"Low disk space ({disk_space}%)")
                
            if memory_usage > 90:
                priority_info["level"] = "high"
                priority_info["urgency"] = "high"
                priority_info["factors"].append(f"Critical memory usage ({memory_usage}%)")
            elif memory_usage > 80:
                priority_info["urgency"] = "medium"
                priority_info["factors"].append(f"High memory usage ({memory_usage}%)")

        # Check software type and importance
        if "software_type" in params:
            software_type = params["software_type"]
            if software_type in ["security", "system", "critical"]:
                priority_info["level"] = "high"
                priority_info["factors"].append(f"Critical software type: {software_type}")

        # Check for security updates
        if params.get("action") == "update" and params.get("update_type") == "security":
            priority_info["level"] = "high"
            priority_info["urgency"] = "high"
            priority_info["factors"].append("Security update required")

        # Determine impact
        priority_info["impact"] = {
            "system_impact": "high" if priority_info["level"] == "high" else "normal",
            "resource_impact": "significant" if priority_info["urgency"] == "high" else "minimal",
            "user_impact": "immediate" if priority_info["urgency"] == "high" else "standard",
            "service_impact": "critical" if priority_info["level"] == "high" else "normal"
        }

        # Set handling recommendations
        priority_info["handling"] = {
            "execution_order": "immediate" if priority_info["urgency"] == "high" else 
                             "next_in_queue" if priority_info["urgency"] == "medium" else 
                             "normal",
            "backup_required": priority_info["level"] == "high",
            "monitoring_level": "intensive" if priority_info["level"] == "high" else "standard",
            "notification_required": priority_info["level"] in ["high", "medium"],
            "rollback_plan": priority_info["level"] == "high"
        }

        return priority_info

    def _determine_strategy(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive strategy for handling the software operation"""
        strategy = {
            "approach": "",
            "steps": [],
            "validations": [],
            "fallback": {},
            "optimization": {},
            "monitoring": {}
        }

        if action == "install":
            strategy.update({
                "approach": "Verify requirements, prepare system, and install efficiently",
                "steps": [
                    "Check system requirements",
                    "Verify dependencies",
                    "Prepare installation space",
                    "Download package",
                    "Install software",
                    "Configure settings",
                    "Verify installation"
                ],
                "validations": [
                    "System compatibility",
                    "Resource availability",
                    "Package integrity",
                    "Installation success"
                ],
                "fallback": {
                    "space_issue": "Clean unnecessary files",
                    "dependency_conflict": "Resolve conflicts",
                    "download_fail": "Try alternate source",
                    "install_error": "Rollback changes"
                },
                "optimization": {
                    "parallel_downloads": True,
                    "compression": "auto",
                    "cache_usage": True,
                    "resource_allocation": "dynamic"
                },
                "monitoring": {
                    "resource_usage": True,
                    "network_activity": True,
                    "system_stability": True,
                    "installation_progress": True
                }
            })
        elif action == "update":
            strategy.update({
                "approach": "Backup current state and update while preserving settings",
                "steps": [
                    "Create backup",
                    "Check dependencies",
                    "Download updates",
                    "Stop services",
                    "Apply updates",
                    "Restart services",
                    "Verify functionality"
                ],
                "validations": [
                    "Backup integrity",
                    "Update compatibility",
                    "Service status",
                    "System stability"
                ],
                "fallback": {
                    "backup_fail": "Skip update",
                    "update_error": "Restore backup",
                    "service_error": "Manual restart",
                    "verification_fail": "Rollback"
                },
                "optimization": {
                    "delta_updates": True,
                    "service_handling": "graceful",
                    "parallel_processing": True,
                    "resource_management": "adaptive"
                },
                "monitoring": {
                    "update_progress": True,
                    "service_health": True,
                    "system_metrics": True,
                    "error_logging": True
                }
            })
        elif action == "uninstall":
            strategy.update({
                "approach": "Remove software and clean up associated files",
                "steps": [
                    "Stop services",
                    "Backup configurations",
                    "Remove components",
                    "Clean dependencies",
                    "Remove data",
                    "Update system",
                    "Verify cleanup"
                ],
                "validations": [
                    "Service status",
                    "Backup success",
                    "Removal completion",
                    "System integrity"
                ],
                "fallback": {
                    "service_running": "Force stop",
                    "removal_error": "Manual cleanup",
                    "dependency_issue": "Skip optional",
                    "system_error": "Restore state"
                },
                "optimization": {
                    "parallel_removal": True,
                    "space_recovery": True,
                    "dependency_handling": "smart",
                    "system_update": "async"
                },
                "monitoring": {
                    "removal_progress": True,
                    "system_stability": True,
                    "space_recovery": True,
                    "dependency_status": True
                }
            })

        return strategy

    def _determine_considerations(self, action: str, params: Dict[str, Any]) -> List[str]:
        """Determine important considerations for the software operation"""
        considerations = []
        
        if action == "install":
            considerations.extend([
                "System requirements",
                "Available disk space",
                "Required dependencies",
                "Installation location"
            ])
        elif action == "update":
            considerations.extend([
                "Current version",
                "Update size",
                "Backup requirements",
                "Compatibility issues"
            ])
        elif action == "uninstall":
            considerations.extend([
                "Associated data",
                "Dependencies",
                "System impact",
                "Cleanup requirements"
            ])
            
        return considerations

    async def _check_compatibility_with_rag(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check software compatibility using RAG"""
        try:
            result = self.rag_skill.execute({
                "action": "generate",
                "query": "check software compatibility",
                "context": params
            })

            if not result["success"]:
                return result

            return {
                "success": True,
                "compatibility": result["content"],
                "requirements": result.get("requirements", {}),
                "reasoning": result.get("reasoning", {})
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"Compatibility check error: {str(e)}"]
            }

    def _update_metrics(self, action: str, success: bool, params: Dict[str, Any]) -> None:
        """Update comprehensive skill metrics based on action and result"""
        start_time = datetime.now()
        
        if success:
            self.metrics["successful_operations"] += 1
            
            # Update action-specific metrics
            if action == "install":
                self.metrics["installations_completed"] += 1
                if "installation_time" in params:
                    install_time = params["installation_time"]
                    self.metrics["total_installation_time"] += install_time
                    # Update average installation time
                    self.metrics["performance_metrics"]["avg_install_time"] = (
                        self.metrics["performance_metrics"]["avg_install_time"] * 
                        (self.metrics["installations_completed"] - 1) +
                        install_time
                    ) / self.metrics["installations_completed"]
                
                # Update package metrics
                if "software_name" in params:
                    name = params["software_name"]
                    self.metrics["package_metrics"]["packages_installed"][name] = (
                        self.metrics["package_metrics"]["packages_installed"].get(name, 0) + 1
                    )
                    if "package_size" in params:
                        self.metrics["package_metrics"]["installation_sizes"][name] = params["package_size"]
                
                # Update dependency metrics
                if "dependencies" in params:
                    deps = params["dependencies"]
                    self.metrics["dependency_metrics"]["dependencies_installed"] += len(deps)
                    for dep in deps:
                        if dep not in self.metrics["dependency_metrics"]["shared_dependencies"]:
                            self.metrics["dependency_metrics"]["shared_dependencies"][dep] = []
                        self.metrics["dependency_metrics"]["shared_dependencies"][dep].append(
                            params["software_name"]
                        )
                
            elif action == "update":
                self.metrics["updates_completed"] += 1
                if "update_time" in params:
                    update_time = params["update_time"]
                    # Update average update time
                    self.metrics["performance_metrics"]["avg_update_time"] = (
                        self.metrics["performance_metrics"]["avg_update_time"] * 
                        (self.metrics["updates_completed"] - 1) +
                        update_time
                    ) / self.metrics["updates_completed"]
                
                if "software_name" in params:
                    name = params["software_name"]
                    self.metrics["package_metrics"]["packages_updated"][name] = (
                        self.metrics["package_metrics"]["packages_updated"].get(name, 0) + 1
                    )
                    if "update_size" in params:
                        self.metrics["package_metrics"]["update_sizes"][name] = params["update_size"]
                
            elif action == "uninstall":
                self.metrics["uninstallations_completed"] += 1
                if "space_freed" in params:
                    self.metrics["disk_space_saved"] += params["space_freed"]
                
                if "software_name" in params:
                    name = params["software_name"]
                    self.metrics["package_metrics"]["packages_removed"][name] = (
                        self.metrics["package_metrics"]["packages_removed"].get(name, 0) + 1
                    )
                
                # Update dependency metrics
                if "orphaned_deps" in params:
                    self.metrics["dependency_metrics"]["orphaned_dependencies"] += params["orphaned_deps"]
                
            elif action == "manage_extension":
                self.metrics["extensions_managed"] += 1
            elif action == "change_settings":
                self.metrics["settings_changed"] += 1
            elif action == "check_compatibility":
                self.metrics["compatibility_checks_run"] += 1

            # Update performance metrics
            if "performance_data" in params:
                perf = params["performance_data"]
                self.metrics["performance_metrics"].update({
                    "peak_cpu_usage": max(
                        self.metrics["performance_metrics"]["peak_cpu_usage"],
                        perf.get("cpu_usage", 0)
                    ),
                    "peak_memory_usage": max(
                        self.metrics["performance_metrics"]["peak_memory_usage"],
                        perf.get("memory_usage", 0)
                    ),
                    "disk_io": perf.get("disk_io", 0),
                    "network_io": perf.get("network_io", 0)
                })

            # Update system impact metrics
            if "system_impact" in params:
                impact = params["system_impact"]
                self.metrics["system_impact"].update({
                    "bootup_time_change": impact.get("bootup_change", 0),
                    "memory_footprint": impact.get("memory_footprint", 0),
                    "cpu_load_impact": impact.get("cpu_impact", 0),
                    "network_impact": impact.get("network_impact", 0)
                })
                if "disk_usage" in impact:
                    self.metrics["system_impact"]["disk_usage_trend"].append({
                        "timestamp": datetime.now().isoformat(),
                        "usage": impact["disk_usage"]
                    })

            # Update security metrics
            if "security_data" in params:
                sec = params["security_data"]
                self.metrics["security_metrics"].update({
                    "verified_packages": self.metrics["security_metrics"]["verified_packages"] + 
                                      sec.get("verified", 0),
                    "unverified_packages": self.metrics["security_metrics"]["unverified_packages"] + 
                                        sec.get("unverified", 0),
                    "security_updates": self.metrics["security_metrics"]["security_updates"] + 
                                     sec.get("security_updates", 0),
                    "vulnerability_patches": self.metrics["security_metrics"]["vulnerability_patches"] + 
                                         sec.get("patches", 0),
                    "signature_validations": self.metrics["security_metrics"]["signature_validations"] + 
                                         sec.get("validations", 0)
                })

            # Update backup metrics
            if "backup_data" in params:
                backup = params["backup_data"]
                self.metrics["backup_metrics"].update({
                    "backups_created": self.metrics["backup_metrics"]["backups_created"] + 
                                    backup.get("created", 0),
                    "backups_restored": self.metrics["backup_metrics"]["backups_restored"] + 
                                    backup.get("restored", 0),
                    "backup_size_total": self.metrics["backup_metrics"]["backup_size_total"] + 
                                     backup.get("size", 0),
                    "successful_rollbacks": self.metrics["backup_metrics"]["successful_rollbacks"] + 
                                       backup.get("successful_rollbacks", 0),
                    "failed_rollbacks": self.metrics["backup_metrics"]["failed_rollbacks"] + 
                                    backup.get("failed_rollbacks", 0)
                })

        else:
            self.metrics["failed_operations"] += 1
            
            # Update error metrics
            if "error_type" in params:
                error_type = params["error_type"]
                if action == "install":
                    self.metrics["error_metrics"]["installation_errors"][error_type] = (
                        self.metrics["error_metrics"]["installation_errors"].get(error_type, 0) + 1
                    )
                elif action == "update":
                    self.metrics["error_metrics"]["update_errors"][error_type] = (
                        self.metrics["error_metrics"]["update_errors"].get(error_type, 0) + 1
                    )
                elif action == "check_compatibility":
                    self.metrics["error_metrics"]["compatibility_errors"][error_type] = (
                        self.metrics["error_metrics"]["compatibility_errors"].get(error_type, 0) + 1
                    )
                
            if "error_category" in params:
                category = params["error_category"]
                if category == "permission":
                    self.metrics["error_metrics"]["permission_errors"] += 1
                elif category == "network":
                    self.metrics["error_metrics"]["network_errors"] += 1
                elif category == "disk_space":
                    self.metrics["error_metrics"]["disk_space_errors"] += 1

    async def _execute_action(self, action: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute software action with context"""
        try:
            if action == "update":
                return await self._update_software(params, context)
            elif action == "restart":
                return await self._restart_software(params, context)
            elif action == "install":
                return await self._install_software(params, context)
            elif action == "uninstall":
                return await self._uninstall_software(params, context)
            elif action == "check_compatibility":
                return await self._check_compatibility(params, context)
            elif action == "setup_auto_update":
                return await self._setup_auto_update(params, context)
            elif action == "manage_extension":
                return await self._manage_extension(params, context)
            elif action == "change_settings":
                return await self._change_settings(params, context)
            elif action == "accept_terms":
                return await self._accept_terms(context)
            elif action == "subscribe_notifications":
                return await self._subscribe_notifications(context)
            elif action == "enter_product_key":
                return await self._enter_product_key(params, context)
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

    async def _update_software(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Update specified software with context"""
        try:
            # Implementation would handle software update
            # For now, returning mock success
            return {
                "success": True,
                "message": "Software updated successfully",
                "context": context
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "context": context
            }

    async def _check_compatibility(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Check software compatibility with context"""
        try:
            # Get system information
            system_info = {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "disk_space": psutil.disk_usage("/").free,
                "os_version": os.uname().version
            }
            
            # Check against requirements
            requirements = context.get("requirements", {})
            compatibility_issues = []
            
            if "min_cpu_cores" in requirements and system_info["cpu_count"] < requirements["min_cpu_cores"]:
                compatibility_issues.append("Insufficient CPU cores")
            
            if "min_memory" in requirements and system_info["memory_total"] < requirements["min_memory"]:
                compatibility_issues.append("Insufficient memory")
            
            if "min_disk_space" in requirements and system_info["disk_space"] < requirements["min_disk_space"]:
                compatibility_issues.append("Insufficient disk space")
            
            return {
                "success": len(compatibility_issues) == 0,
                "message": "Compatibility check completed",
                "system_info": system_info,
                "compatibility_issues": compatibility_issues,
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
