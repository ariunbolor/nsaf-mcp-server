from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import json
from pathlib import Path
from .base_skill import BaseSkill
from .rag_skill import RAGSkill

class DocumentManagementSkill(BaseSkill):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="document_management",
            description="A skill for managing document operations including opening, saving, printing, and formatting documents",
            config=config
        )
        self.required_credentials = [
            "printer_api_key"  # For printer operations
        ]
        self.rag_skill = RAGSkill()
        self.metrics = {
            "documents_opened": 0,
            "documents_saved": 0,
            "pages_printed": 0,
            "pages_scanned": 0,
            "format_changes": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_processing_time": 0,  # in seconds
            "pdf_conversions": 0,
            "images_inserted": 0,
            "spreadsheet_operations": 0,
            "document_metrics": {
                "file_types": {},  # type: count
                "avg_file_size": 0.0,  # in bytes
                "total_pages": 0,
                "word_count": 0,
                "document_languages": {},  # language: count
                "document_categories": {
                    "reports": 0,
                    "letters": 0,
                    "presentations": 0,
                    "spreadsheets": 0,
                    "other": 0
                }
            },
            "formatting_metrics": {
                "style_changes": 0,
                "font_usage": {},  # font: count
                "color_usage": {},  # color: count
                "alignment_changes": 0,
                "spacing_adjustments": 0,
                "template_usage": {},  # template: count
                "style_consistency": 0.0  # percentage
            },
            "printer_metrics": {
                "color_pages": 0,
                "bw_pages": 0,
                "paper_sizes": {},  # size: count
                "duplex_prints": 0,
                "print_quality": {},  # quality: count
                "ink_usage": 0.0,  # percentage
                "paper_usage": 0.0,  # percentage
                "print_errors": {}  # error_type: count
            },
            "scanner_metrics": {
                "scan_quality": {},  # quality: count
                "scan_sizes": {},  # size: count
                "color_scans": 0,
                "bw_scans": 0,
                "ocr_accuracy": 0.0,  # percentage
                "scan_errors": {},  # error_type: count
                "file_formats": {}  # format: count
            },
            "conversion_metrics": {
                "source_formats": {},  # format: count
                "target_formats": {},  # format: count
                "conversion_quality": 0.0,  # percentage
                "format_compatibility": {},  # format_pair: success_rate
                "conversion_errors": {},  # error_type: count
                "size_changes": {}  # format: avg_change_ratio
            },
            "performance_metrics": {
                "avg_open_time": 0.0,  # in seconds
                "avg_save_time": 0.0,  # in seconds
                "avg_print_time": 0.0,  # in seconds
                "avg_scan_time": 0.0,  # in seconds
                "memory_usage": 0.0,  # in MB
                "cpu_usage": 0.0,  # percentage
                "operation_latency": {}  # operation: avg_time
            },
            "spreadsheet_metrics": {
                "cell_operations": 0,
                "formula_usage": {},  # formula_type: count
                "data_types": {},  # type: count
                "sheet_count": 0,
                "data_validation": 0,
                "chart_types": {},  # type: count
                "macro_usage": 0
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters for document operations"""
        errors = []
        required_params = {
            "action": ["open_document", "close_document", "save_document", 
                      "print_document", "scan_document", "edit_document",
                      "copy_paste", "format_document", "insert_image",
                      "convert_pdf", "manage_spreadsheet"],
            "file_path": ["open_document", "save_document", "print_document",
                         "convert_pdf"],
            "format_options": ["format_document"],
            "image_path": ["insert_image"],
            "conversion_format": ["convert_pdf"],
            "spreadsheet_data": ["manage_spreadsheet"]
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
        """Execute document management operations with reasoning"""
        if params is None:
            params = {}

        # Validate parameters
        errors = self.validate_params(params)
        if errors:
            return {"success": False, "errors": errors}

        # For printer operations, check credentials
        if params.get("action") in ["print_document", "scan_document"]:
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
            start_time = datetime.now()
            result = await self._execute_action(action, params, reasoning_result.get("context", {}))
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self._update_metrics(action, result["success"], params, processing_time)
            
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
        """Apply human-like reasoning to document operations"""
        try:
            reasoning_context = {
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "goal": self._determine_goal(action, params),
                "priority": self._determine_priority(params),
                "strategy": self._determine_strategy(action, params),
                "considerations": self._determine_considerations(action, params)
            }

            # For formatting, get recommendations
            if action == "format_document":
                format_result = await self._get_format_recommendations(params)
                if not format_result["success"]:
                    return format_result
                reasoning_context["format_recommendations"] = format_result["recommendations"]
                reasoning_context["style_guide"] = format_result["style_guide"]

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
        """Determine the comprehensive goal of the document operation"""
        goals = {
            "primary_goal": "",
            "sub_goals": [],
            "success_criteria": [],
            "constraints": [],
            "optimization_targets": []
        }

        if action == "format_document":
            goals.update({
                "primary_goal": "Improve document readability and professional appearance",
                "sub_goals": [
                    "Apply consistent styling",
                    "Enhance readability",
                    "Optimize layout",
                    "Ensure accessibility"
                ],
                "success_criteria": [
                    "Style consistency",
                    "Clear formatting",
                    "Professional look",
                    "User satisfaction"
                ],
                "constraints": [
                    "Document type",
                    "Content structure",
                    "Style guidelines",
                    "Format limitations"
                ],
                "optimization_targets": [
                    "Visual appeal",
                    "Reading efficiency",
                    "Style compliance",
                    "Format compatibility"
                ]
            })
        elif action == "print_document":
            goals.update({
                "primary_goal": "Create high-quality physical copy while optimizing resources",
                "sub_goals": [
                    "Configure settings",
                    "Optimize quality",
                    "Manage resources",
                    "Ensure accuracy"
                ],
                "success_criteria": [
                    "Print quality",
                    "Resource efficiency",
                    "Color accuracy",
                    "Paper handling"
                ],
                "constraints": [
                    "Printer capabilities",
                    "Resource availability",
                    "Quality requirements",
                    "Time constraints"
                ],
                "optimization_targets": [
                    "Print speed",
                    "Resource usage",
                    "Output quality",
                    "Cost efficiency"
                ]
            })
        elif action == "convert_pdf":
            goals.update({
                "primary_goal": "Preserve document formatting and ensure compatibility",
                "sub_goals": [
                    "Analyze content",
                    "Convert format",
                    "Preserve layout",
                    "Verify output"
                ],
                "success_criteria": [
                    "Format accuracy",
                    "Content preservation",
                    "Layout integrity",
                    "File compatibility"
                ],
                "constraints": [
                    "Source format",
                    "Target format",
                    "File size",
                    "Quality needs"
                ],
                "optimization_targets": [
                    "Conversion accuracy",
                    "File size",
                    "Processing speed",
                    "Quality retention"
                ]
            })
        elif action == "manage_spreadsheet":
            goals.update({
                "primary_goal": "Organize and analyze data effectively",
                "sub_goals": [
                    "Structure data",
                    "Apply formulas",
                    "Create visualizations",
                    "Enable analysis"
                ],
                "success_criteria": [
                    "Data organization",
                    "Formula accuracy",
                    "Visual clarity",
                    "Analysis capability"
                ],
                "constraints": [
                    "Data volume",
                    "Sheet structure",
                    "Formula complexity",
                    "Performance needs"
                ],
                "optimization_targets": [
                    "Data efficiency",
                    "Calculation speed",
                    "Memory usage",
                    "User interface"
                ]
            })
        else:
            goals.update({
                "primary_goal": "Handle document operation efficiently while maintaining quality",
                "sub_goals": [
                    "Process document",
                    "Maintain quality",
                    "Optimize performance",
                    "Ensure reliability"
                ],
                "success_criteria": [
                    "Operation success",
                    "Quality preservation",
                    "Performance efficiency",
                    "User satisfaction"
                ],
                "constraints": [
                    "System resources",
                    "Time requirements",
                    "Quality standards",
                    "User needs"
                ],
                "optimization_targets": [
                    "Processing speed",
                    "Resource usage",
                    "Output quality",
                    "User experience"
                ]
            })

        return goals

    def _determine_priority(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive priority of the document operation"""
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

        # Check document type and size
        if params.get("file_path"):
            file_size = os.path.getsize(params["file_path"])
            file_type = os.path.splitext(params["file_path"])[1].lower()
            
            # Size-based priority
            if file_size > 100 * 1024 * 1024:  # 100MB
                priority_info["level"] = "high"
                priority_info["urgency"] = "high"
                priority_info["factors"].append("Very large file size")
            elif file_size > 10 * 1024 * 1024:  # 10MB
                priority_info["urgency"] = "medium"
                priority_info["factors"].append("Large file size")

            # Type-based priority
            critical_types = [".pptx", ".key", ".docx", ".xlsx"]
            if file_type in critical_types:
                priority_info["level"] = "high"
                priority_info["factors"].append(f"Critical file type: {file_type}")

        # Check operation type
        action = params.get("action", "")
        if action == "print_document":
            if params.get("urgent", False):
                priority_info["level"] = "high"
                priority_info["urgency"] = "high"
                priority_info["factors"].append("Urgent printing needed")
        elif action == "convert_pdf":
            if params.get("deadline"):
                priority_info["urgency"] = "high"
                priority_info["factors"].append("Conversion deadline approaching")

        # Determine impact
        priority_info["impact"] = {
            "resource_impact": "high" if priority_info["level"] == "high" else "normal",
            "processing_impact": "significant" if priority_info["urgency"] == "high" else "minimal",
            "quality_impact": "critical" if priority_info["level"] == "high" else "standard",
            "time_impact": "immediate" if priority_info["urgency"] == "high" else "normal"
        }

        # Set handling recommendations
        priority_info["handling"] = {
            "execution_order": "immediate" if priority_info["urgency"] == "high" else 
                             "next_in_queue" if priority_info["urgency"] == "medium" else 
                             "normal",
            "resource_allocation": "high" if priority_info["level"] == "high" else "normal",
            "quality_settings": "maximum" if priority_info["level"] == "high" else "standard",
            "backup_required": priority_info["level"] == "high",
            "monitoring_level": "intensive" if priority_info["level"] == "high" else "normal"
        }

        return priority_info

    def _determine_strategy(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive strategy for handling the document operation"""
        strategy = {
            "approach": "",
            "steps": [],
            "validations": [],
            "fallback": {},
            "optimization": {},
            "monitoring": {}
        }

        if action == "format_document":
            strategy.update({
                "approach": "Apply consistent formatting while preserving content",
                "steps": [
                    "Analyze document",
                    "Apply styles",
                    "Check consistency",
                    "Optimize layout",
                    "Verify formatting"
                ],
                "validations": [
                    "Style consistency",
                    "Content integrity",
                    "Layout quality",
                    "Format compliance"
                ],
                "fallback": {
                    "style_conflict": "Use document defaults",
                    "format_error": "Skip problematic section",
                    "layout_issue": "Maintain original",
                    "content_loss": "Preserve content over style"
                },
                "optimization": {
                    "batch_formatting": True,
                    "style_inheritance": True,
                    "layout_optimization": True,
                    "memory_efficient": True
                },
                "monitoring": {
                    "style_consistency": True,
                    "content_preservation": True,
                    "format_quality": True,
                    "performance_impact": True
                }
            })
        elif action == "print_document":
            strategy.update({
                "approach": "Optimize print settings for quality and efficiency",
                "steps": [
                    "Check printer",
                    "Configure settings",
                    "Prepare document",
                    "Send to printer",
                    "Verify output"
                ],
                "validations": [
                    "Printer status",
                    "Resource availability",
                    "Print quality",
                    "Output accuracy"
                ],
                "fallback": {
                    "printer_error": "Use alternate printer",
                    "resource_low": "Optimize usage",
                    "quality_issue": "Adjust settings",
                    "size_mismatch": "Scale to fit"
                },
                "optimization": {
                    "resource_usage": True,
                    "quality_balance": True,
                    "speed_optimization": True,
                    "color_management": True
                },
                "monitoring": {
                    "print_progress": True,
                    "resource_levels": True,
                    "output_quality": True,
                    "error_detection": True
                }
            })
        elif action == "convert_pdf":
            strategy.update({
                "approach": "Ensure accurate conversion with minimal loss",
                "steps": [
                    "Analyze source",
                    "Prepare conversion",
                    "Convert content",
                    "Verify output",
                    "Optimize result"
                ],
                "validations": [
                    "Content accuracy",
                    "Format integrity",
                    "Quality level",
                    "File compatibility"
                ],
                "fallback": {
                    "conversion_error": "Try alternate method",
                    "quality_loss": "Increase settings",
                    "size_issue": "Compress result",
                    "format_mismatch": "Use intermediate format"
                },
                "optimization": {
                    "quality_preservation": True,
                    "size_optimization": True,
                    "format_compatibility": True,
                    "processing_efficiency": True
                },
                "monitoring": {
                    "conversion_progress": True,
                    "quality_metrics": True,
                    "resource_usage": True,
                    "error_handling": True
                }
            })

        return strategy

    def _determine_considerations(self, action: str, params: Dict[str, Any]) -> List[str]:
        """Determine important considerations for the document operation"""
        considerations = []
        
        if action == "format_document":
            considerations.extend([
                "Document type",
                "Content structure",
                "Style consistency",
                "Reader accessibility"
            ])
        elif action == "print_document":
            considerations.extend([
                "Paper size",
                "Color requirements",
                "Print quality",
                "Resource usage"
            ])
        elif action == "convert_pdf":
            considerations.extend([
                "Original format",
                "Content preservation",
                "File size",
                "Compatibility"
            ])
            
        return considerations

    async def _get_format_recommendations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get document formatting recommendations using RAG"""
        try:
            result = self.rag_skill.execute({
                "action": "generate",
                "query": "document formatting recommendations",
                "context": params
            })

            if not result["success"]:
                return result

            return {
                "success": True,
                "recommendations": result["content"],
                "style_guide": result.get("style_guide", {}),
                "reasoning": result.get("reasoning", {})
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"Format recommendation error: {str(e)}"]
            }

    def _update_metrics(self, action: str, success: bool, params: Dict[str, Any], processing_time: float) -> None:
        """Update comprehensive skill metrics based on action and result"""
        if success:
            self.metrics["successful_operations"] += 1
            self.metrics["total_processing_time"] += processing_time
            
            # Update document metrics
            if "file_path" in params:
                file_type = os.path.splitext(params["file_path"])[1].lower()
                self.metrics["document_metrics"]["file_types"][file_type] = (
                    self.metrics["document_metrics"]["file_types"].get(file_type, 0) + 1
                )
                
                file_size = os.path.getsize(params["file_path"])
                doc_count = sum(self.metrics["document_metrics"]["file_types"].values())
                self.metrics["document_metrics"]["avg_file_size"] = (
                    (self.metrics["document_metrics"]["avg_file_size"] * (doc_count - 1) + 
                     file_size) / doc_count
                )
            
            # Update action-specific metrics
            if action == "open_document":
                self.metrics["documents_opened"] += 1
                self.metrics["performance_metrics"]["avg_open_time"] = (
                    (self.metrics["performance_metrics"]["avg_open_time"] * 
                     (self.metrics["documents_opened"] - 1) + processing_time) /
                    self.metrics["documents_opened"]
                )
                
            elif action == "save_document":
                self.metrics["documents_saved"] += 1
                self.metrics["performance_metrics"]["avg_save_time"] = (
                    (self.metrics["performance_metrics"]["avg_save_time"] * 
                     (self.metrics["documents_saved"] - 1) + processing_time) /
                    self.metrics["documents_saved"]
                )
                
            elif action == "print_document":
                if "pages" in params:
                    pages = params["pages"]
                    self.metrics["pages_printed"] += pages
                    
                    # Update printer metrics
                    if "color_mode" in params:
                        if params["color_mode"] == "color":
                            self.metrics["printer_metrics"]["color_pages"] += pages
                        else:
                            self.metrics["printer_metrics"]["bw_pages"] += pages
                    
                    if "paper_size" in params:
                        size = params["paper_size"]
                        self.metrics["printer_metrics"]["paper_sizes"][size] = (
                            self.metrics["printer_metrics"]["paper_sizes"].get(size, 0) + pages
                        )
                    
                    if "duplex" in params and params["duplex"]:
                        self.metrics["printer_metrics"]["duplex_prints"] += pages // 2
                    
                    self.metrics["performance_metrics"]["avg_print_time"] = (
                        (self.metrics["performance_metrics"]["avg_print_time"] * 
                         (self.metrics["pages_printed"] - pages) + processing_time) /
                        self.metrics["pages_printed"]
                    )
                
            elif action == "scan_document":
                if "pages" in params:
                    pages = params["pages"]
                    self.metrics["pages_scanned"] += pages
                    
                    # Update scanner metrics
                    if "quality" in params:
                        quality = params["quality"]
                        self.metrics["scanner_metrics"]["scan_quality"][quality] = (
                            self.metrics["scanner_metrics"]["scan_quality"].get(quality, 0) + 1
                        )
                    
                    if "color_mode" in params:
                        if params["color_mode"] == "color":
                            self.metrics["scanner_metrics"]["color_scans"] += pages
                        else:
                            self.metrics["scanner_metrics"]["bw_scans"] += pages
                    
                    if "ocr_accuracy" in params:
                        current_accuracy = self.metrics["scanner_metrics"]["ocr_accuracy"]
                        self.metrics["scanner_metrics"]["ocr_accuracy"] = (
                            (current_accuracy * (self.metrics["pages_scanned"] - pages) +
                             params["ocr_accuracy"] * pages) /
                            self.metrics["pages_scanned"]
                        )
                    
                    self.metrics["performance_metrics"]["avg_scan_time"] = (
                        (self.metrics["performance_metrics"]["avg_scan_time"] * 
                         (self.metrics["pages_scanned"] - pages) + processing_time) /
                        self.metrics["pages_scanned"]
                    )
                
            elif action == "format_document":
                self.metrics["format_changes"] += 1
                
                # Update formatting metrics
                if "format_options" in params:
                    options = params["format_options"]
                    if "font_family" in options:
                        font = options["font_family"]
                        self.metrics["formatting_metrics"]["font_usage"][font] = (
                            self.metrics["formatting_metrics"]["font_usage"].get(font, 0) + 1
                        )
                    
                    if "alignment" in options:
                        self.metrics["formatting_metrics"]["alignment_changes"] += 1
                    
                    if "line_spacing" in options:
                        self.metrics["formatting_metrics"]["spacing_adjustments"] += 1
                
            elif action == "insert_image":
                self.metrics["images_inserted"] += 1
                
            elif action == "convert_pdf":
                self.metrics["pdf_conversions"] += 1
                
                # Update conversion metrics
                if "source_format" in params and "target_format" in params:
                    source = params["source_format"]
                    target = params["target_format"]
                    
                    self.metrics["conversion_metrics"]["source_formats"][source] = (
                        self.metrics["conversion_metrics"]["source_formats"].get(source, 0) + 1
                    )
                    self.metrics["conversion_metrics"]["target_formats"][target] = (
                        self.metrics["conversion_metrics"]["target_formats"].get(target, 0) + 1
                    )
                    
                    format_pair = f"{source}_to_{target}"
                    if "conversion_success" in params:
                        current_rate = self.metrics["conversion_metrics"]["format_compatibility"].get(format_pair, 0)
                        success_count = self.metrics["conversion_metrics"]["source_formats"][source]
                        self.metrics["conversion_metrics"]["format_compatibility"][format_pair] = (
                            (current_rate * (success_count - 1) + 
                             (1 if params["conversion_success"] else 0)) / success_count
                        )
                
            elif action == "manage_spreadsheet":
                self.metrics["spreadsheet_operations"] += 1
                
                # Update spreadsheet metrics
                if "spreadsheet_data" in params:
                    data = params["spreadsheet_data"]
                    if "formulas" in data:
                        for formula_type in data["formulas"]:
                            self.metrics["spreadsheet_metrics"]["formula_usage"][formula_type] = (
                                self.metrics["spreadsheet_metrics"]["formula_usage"].get(formula_type, 0) + 1
                            )
                    
                    if "sheets" in data:
                        self.metrics["spreadsheet_metrics"]["sheet_count"] += len(data["sheets"])
                    
                    if "charts" in data:
                        for chart_type in data["charts"]:
                            self.metrics["spreadsheet_metrics"]["chart_types"][chart_type] = (
                                self.metrics["spreadsheet_metrics"]["chart_types"].get(chart_type, 0) + 1
                            )
            
            # Update performance metrics
            if "resource_usage" in params:
                usage = params["resource_usage"]
                self.metrics["performance_metrics"]["memory_usage"] = max(
                    self.metrics["performance_metrics"]["memory_usage"],
                    usage.get("memory", 0)
                )
                self.metrics["performance_metrics"]["cpu_usage"] = max(
                    self.metrics["performance_metrics"]["cpu_usage"],
                    usage.get("cpu", 0)
                )
            
            # Update operation latency
            if action in self.metrics["performance_metrics"]["operation_latency"]:
                current_avg = self.metrics["performance_metrics"]["operation_latency"][action]
                op_count = self.metrics["successful_operations"]
                self.metrics["performance_metrics"]["operation_latency"][action] = (
                    (current_avg * (op_count - 1) + processing_time) / op_count
                )
            else:
                self.metrics["performance_metrics"]["operation_latency"][action] = processing_time
            
        else:
            self.metrics["failed_operations"] += 1
            
            # Update error metrics for specific actions
            if action == "print_document":
                if "error_type" in params:
                    error_type = params["error_type"]
                    self.metrics["printer_metrics"]["print_errors"][error_type] = (
                        self.metrics["printer_metrics"]["print_errors"].get(error_type, 0) + 1
                    )
            elif action == "scan_document":
                if "error_type" in params:
                    error_type = params["error_type"]
                    self.metrics["scanner_metrics"]["scan_errors"][error_type] = (
                        self.metrics["scanner_metrics"]["scan_errors"].get(error_type, 0) + 1
                    )
            elif action == "convert_pdf":
                if "error_type" in params:
                    error_type = params["error_type"]
                    self.metrics["conversion_metrics"]["conversion_errors"][error_type] = (
                        self.metrics["conversion_metrics"]["conversion_errors"].get(error_type, 0) + 1
                    )

    async def _execute_action(self, action: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document action with context"""
        try:
            if action == "open_document":
                return await self._open_document(params, context)
            elif action == "close_document":
                return await self._close_document(context)
            elif action == "save_document":
                return await self._save_document(params, context)
            elif action == "print_document":
                return await self._print_document(params, context)
            elif action == "scan_document":
                return await self._scan_document(context)
            elif action == "edit_document":
                return await self._edit_document(params, context)
            elif action == "copy_paste":
                return await self._copy_paste(params, context)
            elif action == "format_document":
                return await self._format_document(params, context)
            elif action == "insert_image":
                return await self._insert_image(params, context)
            elif action == "convert_pdf":
                return await self._convert_pdf(params, context)
            elif action == "manage_spreadsheet":
                return await self._manage_spreadsheet(params, context)
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

    async def _format_document(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Format document with context"""
        try:
            # Get format options
            format_options = params.get("format_options", {})
            default_formats = self.get_config("settings.default_formats", {})
            
            # Apply formatting with recommendations
            recommendations = context.get("format_recommendations", {})
            style_guide = context.get("style_guide", {})
            
            # Merge format options with recommendations and defaults
            final_format = {
                "font_family": format_options.get("font_family") or 
                             recommendations.get("font_family") or 
                             default_formats.get("font_family", "Arial"),
                "font_size": format_options.get("font_size") or 
                           recommendations.get("font_size") or 
                           default_formats.get("font_size", 12),
                "line_spacing": format_options.get("line_spacing") or 
                              recommendations.get("line_spacing") or 
                              default_formats.get("line_spacing", 1.15)
            }
            
            return {
                "success": True,
                "message": "Document formatted successfully",
                "applied_format": final_format,
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
