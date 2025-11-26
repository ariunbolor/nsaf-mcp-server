from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import shutil
import glob
import psutil
from pathlib import Path
from .base_skill import BaseSkill
from .rag_skill import RAGSkill

class FileOperationsSkill(BaseSkill):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="file_operations",
            description="A skill for managing file operations including organizing, moving, and maintaining files",
            config=config
        )
        self.required_credentials = [
            "cloud_storage_api_key"  # For cloud backup operations
        ]
        self.rag_skill = RAGSkill()
        self.metrics = {
            "files_processed": 0,
            "space_saved": 0,  # in bytes
            "files_organized": 0,
            "successful_backups": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_storage_saved": 0,  # in bytes
            "storage_usage": 0,  # percentage
            "files_moved": 0,
            "files_renamed": 0,
            "files_deleted": 0,
            "folders_created": 0,
            "files_compressed": 0,
            "trash_emptied": 0,
            "searches_performed": 0,
            "storage_checks": 0,
            "file_categories": {
                "documents": 0,
                "images": 0,
                "videos": 0,
                "archives": 0,
                "other": 0
            },
            "organization_patterns": {
                "by_extension": 0,
                "by_date": 0,
                "by_size": 0,
                "by_type": 0,
                "custom": 0
            },
            "storage_metrics": {
                "total_space": 0,  # in bytes
                "used_space": 0,   # in bytes
                "free_space": 0,   # in bytes
                "compression_ratio": 0.0,
                "backup_size": 0,  # in bytes
                "duplicate_files": 0
            },
            "performance_metrics": {
                "avg_operation_time": 0.0,  # in seconds
                "peak_disk_usage": 0,  # percentage
                "successful_rate": 0.0,  # percentage
                "compression_time": 0.0  # in seconds
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters for file operations"""
        errors = []
        required_params = {
            "action": ["rename", "move", "delete", "organize", "create_folder",
                      "compress", "backup", "empty_trash", "search", "check_storage"],
            "file_path": ["rename", "move", "delete", "compress", "backup"],
            "new_name": ["rename"],
            "destination": ["move", "backup"],
            "folder_name": ["create_folder"],
            "search_query": ["search"],
            "file_pattern": ["organize"]
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
        """Execute file operations with reasoning"""
        if params is None:
            params = {}

        # Validate parameters
        errors = self.validate_params(params)
        if errors:
            return {"success": False, "errors": errors}

        # For cloud operations, check credentials
        if params.get("action") == "backup":
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
        """Apply human-like reasoning to file operations"""
        try:
            reasoning_context = {
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "goal": self._determine_goal(action, params),
                "priority": self._determine_priority(params),
                "strategy": self._determine_strategy(action, params),
                "considerations": self._determine_considerations(action, params)
            }

            # For organize action, determine organization pattern
            if action == "organize" and "file_pattern" not in params:
                pattern_result = await self._determine_organization_pattern(params)
                if not pattern_result["success"]:
                    return pattern_result
                params["file_pattern"] = pattern_result["pattern"]
                reasoning_context["organization_pattern"] = pattern_result["reasoning"]

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
        """Determine the comprehensive goal of the file operation"""
        goals = {
            "primary_goal": "",
            "sub_goals": [],
            "success_criteria": [],
            "constraints": [],
            "optimization_targets": []
        }

        if action == "organize":
            goals.update({
                "primary_goal": "Create a logical and efficient file structure",
                "sub_goals": [
                    "Analyze file patterns",
                    "Group similar files",
                    "Create folder hierarchy",
                    "Maintain accessibility"
                ],
                "success_criteria": [
                    "Files properly categorized",
                    "Clear folder structure",
                    "Easy navigation",
                    "Improved findability"
                ],
                "constraints": [
                    "Existing file access patterns",
                    "Storage limitations",
                    "File naming conventions",
                    "System restrictions"
                ],
                "optimization_targets": [
                    "Access efficiency",
                    "Storage utilization",
                    "Organization clarity",
                    "Maintenance ease"
                ]
            })
        elif action == "backup":
            goals.update({
                "primary_goal": "Ensure data safety and redundancy",
                "sub_goals": [
                    "Verify data integrity",
                    "Optimize backup size",
                    "Ensure completeness",
                    "Enable easy recovery"
                ],
                "success_criteria": [
                    "All critical files backed up",
                    "Backup verified",
                    "Storage optimized",
                    "Recovery tested"
                ],
                "constraints": [
                    "Available storage space",
                    "Network bandwidth",
                    "Time limitations",
                    "Security requirements"
                ],
                "optimization_targets": [
                    "Backup speed",
                    "Storage efficiency",
                    "Data integrity",
                    "Recovery time"
                ]
            })
        elif action == "compress":
            goals.update({
                "primary_goal": "Optimize storage space while maintaining accessibility",
                "sub_goals": [
                    "Analyze file content",
                    "Choose compression method",
                    "Preserve metadata",
                    "Ensure recoverability"
                ],
                "success_criteria": [
                    "Space reduction achieved",
                    "File integrity maintained",
                    "Quick access possible",
                    "Format compatibility"
                ],
                "constraints": [
                    "File type limitations",
                    "Processing power",
                    "Time constraints",
                    "Format requirements"
                ],
                "optimization_targets": [
                    "Compression ratio",
                    "Processing speed",
                    "Access performance",
                    "Format support"
                ]
            })
        else:
            goals.update({
                "primary_goal": "Manage files efficiently and maintain system organization",
                "sub_goals": [
                    "Execute operation safely",
                    "Maintain organization",
                    "Track changes",
                    "Optimize storage"
                ],
                "success_criteria": [
                    "Operation completed",
                    "System organized",
                    "Changes tracked",
                    "Space optimized"
                ],
                "constraints": [
                    "System resources",
                    "Time efficiency",
                    "Data integrity",
                    "User permissions"
                ],
                "optimization_targets": [
                    "Operation speed",
                    "Resource usage",
                    "System health",
                    "User experience"
                ]
            })

        return goals

    def _determine_priority(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive priority of the file operation"""
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

        # Check storage space for storage-related operations
        if params.get("action") in ["delete", "compress", "backup"]:
            usage = psutil.disk_usage("/").percent
            threshold = self.get_config("settings.storage_threshold", 90)
            
            if usage > threshold:
                priority_info["level"] = "high"
                priority_info["urgency"] = "high"
                priority_info["factors"].append(f"Storage usage ({usage}%) exceeds threshold ({threshold}%)")
            elif usage > threshold * 0.8:
                priority_info["urgency"] = "medium"
                priority_info["factors"].append(f"Storage usage ({usage}%) approaching threshold")

        # Check file importance if available
        if "file_importance" in params:
            importance = params["file_importance"]
            if importance in ["critical", "high"]:
                priority_info["level"] = "high"
                priority_info["factors"].append(f"File importance: {importance}")

        # Check operation type
        action = params.get("action", "")
        if action == "backup" and priority_info["level"] == "high":
            priority_info["urgency"] = "high"
            priority_info["factors"].append("Critical backup operation")
        elif action == "delete" and priority_info["urgency"] == "high":
            priority_info["level"] = "high"
            priority_info["factors"].append("Urgent storage cleanup needed")

        # Determine impact
        priority_info["impact"] = {
            "storage_impact": "high" if priority_info["urgency"] == "high" else "normal",
            "system_impact": "significant" if priority_info["level"] == "high" else "minimal",
            "user_impact": "immediate" if priority_info["urgency"] == "high" else "standard"
        }

        # Set handling recommendations
        priority_info["handling"] = {
            "execution_order": "immediate" if priority_info["urgency"] == "high" else 
                             "next_in_queue" if priority_info["urgency"] == "medium" else 
                             "normal",
            "verification": priority_info["level"] == "high",
            "backup_first": action == "delete" and priority_info["level"] == "high",
            "monitoring": priority_info["level"] in ["high", "medium"]
        }

        return priority_info

    def _determine_strategy(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive strategy for handling the file operation"""
        strategy = {
            "approach": "",
            "steps": [],
            "validations": [],
            "fallback": {},
            "optimization": {},
            "monitoring": {}
        }

        if action == "organize":
            strategy.update({
                "approach": "Group files logically while maintaining accessibility",
                "steps": [
                    "Analyze file patterns",
                    "Create folder structure",
                    "Move files",
                    "Update references",
                    "Verify organization"
                ],
                "validations": [
                    "File accessibility",
                    "Pattern consistency",
                    "Reference integrity",
                    "Storage efficiency"
                ],
                "fallback": {
                    "pattern_mismatch": "Use default categorization",
                    "space_issue": "Clean up first",
                    "naming_conflict": "Add unique identifier"
                },
                "optimization": {
                    "batch_operations": True,
                    "parallel_processing": True,
                    "incremental_updates": True
                },
                "monitoring": {
                    "track_moves": True,
                    "verify_access": True,
                    "measure_efficiency": True
                }
            })
        elif action == "backup":
            strategy.update({
                "approach": "Ensure data integrity and verify backup success",
                "steps": [
                    "Calculate required space",
                    "Prepare destination",
                    "Copy files",
                    "Verify integrity",
                    "Update records"
                ],
                "validations": [
                    "Space availability",
                    "File integrity",
                    "Backup completeness",
                    "Access permissions"
                ],
                "fallback": {
                    "space_insufficient": "Use compression",
                    "network_error": "Retry with backoff",
                    "verification_fail": "Retry backup"
                },
                "optimization": {
                    "incremental_backup": True,
                    "compression": "auto",
                    "deduplication": True
                },
                "monitoring": {
                    "track_progress": True,
                    "verify_checksums": True,
                    "log_operations": True
                }
            })
        elif action == "compress":
            strategy.update({
                "approach": "Balance compression ratio with processing time",
                "steps": [
                    "Analyze file type",
                    "Choose algorithm",
                    "Apply compression",
                    "Verify result",
                    "Update metadata"
                ],
                "validations": [
                    "File compatibility",
                    "Space savings",
                    "Access speed",
                    "Data integrity"
                ],
                "fallback": {
                    "incompatible_type": "Skip file",
                    "low_savings": "Use alternate method",
                    "processing_timeout": "Reduce quality"
                },
                "optimization": {
                    "adaptive_compression": True,
                    "parallel_processing": True,
                    "memory_efficient": True
                },
                "monitoring": {
                    "track_ratio": True,
                    "measure_speed": True,
                    "verify_quality": True
                }
            })

        return strategy

    def _determine_considerations(self, action: str, params: Dict[str, Any]) -> List[str]:
        """Determine important considerations for the file operation"""
        considerations = []
        
        if action == "organize":
            considerations.extend([
                "File types and extensions",
                "Creation/modification dates",
                "File sizes",
                "Access patterns"
            ])
        elif action == "backup":
            considerations.extend([
                "Available storage space",
                "Network bandwidth",
                "Data sensitivity",
                "Backup frequency"
            ])
        elif action == "compress":
            considerations.extend([
                "File types",
                "Compression ratio",
                "Processing time",
                "Required accessibility"
            ])
            
        return considerations

    async def _determine_organization_pattern(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine file organization pattern using RAG"""
        try:
            result = self.rag_skill.execute({
                "action": "generate",
                "query": "determine file organization pattern",
                "context": params
            })

            if not result["success"]:
                return result

            return {
                "success": True,
                "pattern": result["content"],
                "reasoning": result.get("reasoning", {})
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"Pattern determination error: {str(e)}"]
            }

    def _update_metrics(self, action: str, success: bool, params: Dict[str, Any]) -> None:
        """Update comprehensive skill metrics based on action and result"""
        start_time = datetime.now()
        
        if success:
            self.metrics["successful_operations"] += 1
            self.metrics["files_processed"] += 1
            
            # Update action-specific metrics
            if action == "organize":
                self.metrics["files_organized"] += 1
                if "organization_type" in params:
                    org_type = params["organization_type"]
                    if org_type in self.metrics["organization_patterns"]:
                        self.metrics["organization_patterns"][org_type] += 1
                    
            elif action == "compress":
                original_size = os.path.getsize(params["file_path"])
                compressed_size = os.path.getsize(params["destination"])
                space_saved = original_size - compressed_size
                self.metrics["space_saved"] += space_saved
                self.metrics["total_storage_saved"] += space_saved
                self.metrics["files_compressed"] += 1
                
                # Update compression metrics
                self.metrics["storage_metrics"]["compression_ratio"] = (
                    (self.metrics["storage_metrics"]["compression_ratio"] * 
                     (self.metrics["files_compressed"] - 1) +
                     (compressed_size / original_size)) / 
                    self.metrics["files_compressed"]
                )
                
            elif action == "backup":
                self.metrics["successful_backups"] += 1
                if "backup_size" in params:
                    self.metrics["storage_metrics"]["backup_size"] += params["backup_size"]
                    
            elif action == "move":
                self.metrics["files_moved"] += 1
            elif action == "rename":
                self.metrics["files_renamed"] += 1
            elif action == "delete":
                self.metrics["files_deleted"] += 1
            elif action == "create_folder":
                self.metrics["folders_created"] += 1
            elif action == "empty_trash":
                self.metrics["trash_emptied"] += 1
            elif action == "search":
                self.metrics["searches_performed"] += 1
            elif action == "check_storage":
                self.metrics["storage_checks"] += 1

            # Update file categorization if available
            if "file_type" in params:
                file_type = params["file_type"]
                if file_type in self.metrics["file_categories"]:
                    self.metrics["file_categories"][file_type] += 1
                else:
                    self.metrics["file_categories"]["other"] += 1

            # Update storage metrics
            disk_usage = psutil.disk_usage("/")
            self.metrics["storage_metrics"].update({
                "total_space": disk_usage.total,
                "used_space": disk_usage.used,
                "free_space": disk_usage.free
            })
            self.metrics["storage_usage"] = disk_usage.percent
            
            # Update peak usage if current is higher
            if disk_usage.percent > self.metrics["performance_metrics"]["peak_disk_usage"]:
                self.metrics["performance_metrics"]["peak_disk_usage"] = disk_usage.percent

            # Update performance metrics
            operation_time = (datetime.now() - start_time).total_seconds()
            if action == "compress":
                self.metrics["performance_metrics"]["compression_time"] = (
                    self.metrics["performance_metrics"]["compression_time"] * 0.8 +
                    operation_time * 0.2
                )
            
            # Update average operation time
            self.metrics["performance_metrics"]["avg_operation_time"] = (
                self.metrics["performance_metrics"]["avg_operation_time"] * 0.8 +
                operation_time * 0.2
            )
            
            # Update success rate
            total_ops = self.metrics["successful_operations"] + self.metrics["failed_operations"]
            self.metrics["performance_metrics"]["successful_rate"] = (
                self.metrics["successful_operations"] / total_ops * 100
            )

        else:
            self.metrics["failed_operations"] += 1
            
            # Update success rate on failure
            total_ops = self.metrics["successful_operations"] + self.metrics["failed_operations"]
            self.metrics["performance_metrics"]["successful_rate"] = (
                self.metrics["successful_operations"] / total_ops * 100
            )

    async def _execute_action(self, action: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file action with context"""
        try:
            if action == "rename":
                return await self._rename_file(params, context)
            elif action == "move":
                return await self._move_file(params, context)
            elif action == "delete":
                return await self._delete_file(params, context)
            elif action == "organize":
                return await self._organize_files(params, context)
            elif action == "create_folder":
                return await self._create_folder(params, context)
            elif action == "compress":
                return await self._compress_file(params, context)
            elif action == "backup":
                return await self._backup_file(params, context)
            elif action == "empty_trash":
                return await self._empty_trash(context)
            elif action == "search":
                return await self._search_files(params, context)
            elif action == "check_storage":
                return await self._check_storage(context)
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

    async def _rename_file(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Rename a file with context"""
        try:
            old_path = params["file_path"]
            new_path = os.path.join(os.path.dirname(old_path), params["new_name"])
            
            # Check if destination exists
            if os.path.exists(new_path):
                return {
                    "success": False,
                    "warning": "Destination file already exists",
                    "context": context
                }
            
            os.rename(old_path, new_path)
            return {
                "success": True,
                "message": "File renamed successfully",
                "context": context
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "context": context
            }

    async def _check_storage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check available storage space with context"""
        try:
            disk_usage = psutil.disk_usage("/")
            threshold = self.get_config("settings.storage_threshold", 90)
            
            storage_info = {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent": disk_usage.percent
            }
            
            warning = None
            if disk_usage.percent > threshold:
                warning = f"Storage usage ({disk_usage.percent}%) exceeds threshold ({threshold}%)"
            
            return {
                "success": True,
                "message": "Storage check completed",
                "storage_info": storage_info,
                "warning": warning,
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
