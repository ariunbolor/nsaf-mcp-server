from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re
from .base_skill import BaseSkill
from .rag_skill import RAGSkill

class MiscellaneousTasksSkill(BaseSkill):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="miscellaneous_tasks",
            description="A skill for managing miscellaneous computer tasks including security, networking, and system customization",
            config=config
        )
        self.required_credentials = [
            "admin_password"  # For system-level operations
        ]
        self.rag_skill = RAGSkill()
        self.metrics = {
            "passwords_changed": 0,
            "2fa_setups": 0,
            "wifi_checks": 0,
            "notifications_managed": 0,
            "searches_performed": 0,
            "accounts_managed": 0,
            "tutorials_watched": 0,
            "popups_closed": 0,
            "forms_filled": 0,
            "shortcuts_customized": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_processing_time": 0,  # in seconds
            "security_score": 100  # percentage
        }

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters for miscellaneous operations"""
        errors = []
        required_params = {
            "action": ["change_password", "setup_2fa", "check_wifi", 
                      "manage_notifications", "search_info", "manage_accounts",
                      "watch_tutorial", "close_popup", "fill_form",
                      "customize_shortcuts"],
            "password_data": ["change_password"],
            "wifi_settings": ["check_wifi"],
            "notification_settings": ["manage_notifications"],
            "search_query": ["search_info"],
            "account_data": ["manage_accounts"],
            "form_data": ["fill_form"],
            "shortcut_data": ["customize_shortcuts"]
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
        """Execute miscellaneous operations with reasoning"""
        if params is None:
            params = {}

        # Validate parameters
        errors = self.validate_params(params)
        if errors:
            return {"success": False, "errors": errors}

        # For system operations, check credentials
        if params.get("action") in ["change_password", "setup_2fa", "check_wifi"]:
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
        """Apply human-like reasoning to miscellaneous operations"""
        try:
            reasoning_context = {
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "goal": self._determine_goal(action, params),
                "priority": self._determine_priority(params),
                "strategy": self._determine_strategy(action, params),
                "considerations": self._determine_considerations(action, params)
            }

            # For security operations, get analysis
            if action in ["change_password", "setup_2fa"]:
                security_result = await self._analyze_security(params)
                if not security_result["success"]:
                    return security_result
                reasoning_context["security_analysis"] = security_result["analysis"]
                reasoning_context["recommendations"] = security_result["recommendations"]

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

    def _determine_goal(self, action: str, params: Dict[str, Any]) -> str:
        """Determine the goal of the miscellaneous operation"""
        if action == "change_password":
            return "Enhance account security while ensuring password usability"
        elif action == "setup_2fa":
            return "Implement additional security layer with user convenience"
        elif action == "check_wifi":
            return "Verify and optimize network connectivity"
        elif action == "fill_form":
            return "Complete form accurately and efficiently"
        return "Execute task while optimizing user experience and security"

    def _determine_priority(self, params: Dict[str, Any]) -> str:
        """Determine the priority of the miscellaneous operation"""
        if "priority" in params:
            return params["priority"]
        
        # Check operation type for priority
        if params.get("action") in ["change_password", "setup_2fa", "check_wifi"]:
            return "high"
        return "normal"

    def _determine_strategy(self, action: str, params: Dict[str, Any]) -> str:
        """Determine the strategy for handling the miscellaneous operation"""
        if action == "change_password":
            return "Ensure password strength while maintaining memorability"
        elif action == "check_wifi":
            return "Analyze network status and optimize connection"
        elif action == "fill_form":
            return "Complete form fields accurately and efficiently"
        return "Execute operation while balancing security and usability"

    def _determine_considerations(self, action: str, params: Dict[str, Any]) -> List[str]:
        """Determine important considerations for the miscellaneous operation"""
        considerations = []
        
        if action == "change_password":
            considerations.extend([
                "Password complexity",
                "Previous breaches",
                "Common patterns",
                "Memorability"
            ])
        elif action == "check_wifi":
            considerations.extend([
                "Signal strength",
                "Network security",
                "Bandwidth usage",
                "Interference"
            ])
        elif action == "fill_form":
            considerations.extend([
                "Required fields",
                "Data accuracy",
                "Privacy concerns",
                "Validation rules"
            ])
            
        return considerations

    async def _analyze_security(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security aspects using RAG"""
        try:
            result = self.rag_skill.execute({
                "action": "generate",
                "query": "analyze security requirements",
                "context": params
            })

            if not result["success"]:
                return result

            return {
                "success": True,
                "analysis": result["content"],
                "recommendations": result.get("recommendations", {}),
                "reasoning": result.get("reasoning", {})
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"Security analysis error: {str(e)}"]
            }

    def _update_metrics(self, action: str, success: bool, params: Dict[str, Any], processing_time: float) -> None:
        """Update skill metrics based on action and result"""
        if success:
            self.metrics["successful_operations"] += 1
            self.metrics["total_processing_time"] += processing_time
            
            if action == "change_password":
                self.metrics["passwords_changed"] += 1
                # Update security score based on password strength
                if params.get("password_data", {}).get("new_password"):
                    strength = self._calculate_password_strength(
                        params["password_data"]["new_password"]
                    )
                    self.metrics["security_score"] = (
                        self.metrics["security_score"] * 0.8 + strength * 0.2
                    )
            elif action == "setup_2fa":
                self.metrics["2fa_setups"] += 1
                self.metrics["security_score"] = min(100, self.metrics["security_score"] + 5)
            elif action == "check_wifi":
                self.metrics["wifi_checks"] += 1
            elif action == "manage_notifications":
                self.metrics["notifications_managed"] += 1
            elif action == "search_info":
                self.metrics["searches_performed"] += 1
            elif action == "manage_accounts":
                self.metrics["accounts_managed"] += 1
            elif action == "watch_tutorial":
                self.metrics["tutorials_watched"] += 1
            elif action == "close_popup":
                self.metrics["popups_closed"] += 1
            elif action == "fill_form":
                self.metrics["forms_filled"] += 1
            elif action == "customize_shortcuts":
                self.metrics["shortcuts_customized"] += 1
        else:
            self.metrics["failed_operations"] += 1

    def _calculate_password_strength(self, password: str) -> float:
        """Calculate password strength score (0-100)"""
        score = 0
        
        # Length
        if len(password) >= 12:
            score += 25
        elif len(password) >= 8:
            score += 15
            
        # Complexity
        if re.search(r"[A-Z]", password):  # uppercase
            score += 10
        if re.search(r"[a-z]", password):  # lowercase
            score += 10
        if re.search(r"\d", password):     # digits
            score += 10
        if re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):  # special chars
            score += 10
            
        # Variety
        unique_chars = len(set(password))
        score += min(25, unique_chars * 2)
        
        return min(100, score)

    async def _execute_action(self, action: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute miscellaneous action with context"""
        try:
            if action == "change_password":
                return await self._change_password(params, context)
            elif action == "setup_2fa":
                return await self._setup_2fa(context)
            elif action == "check_wifi":
                return await self._check_wifi(params, context)
            elif action == "manage_notifications":
                return await self._manage_notifications(params, context)
            elif action == "search_info":
                return await self._search_info(params, context)
            elif action == "manage_accounts":
                return await self._manage_accounts(params, context)
            elif action == "watch_tutorial":
                return await self._watch_tutorial(context)
            elif action == "close_popup":
                return await self._close_popup(context)
            elif action == "fill_form":
                return await self._fill_form(params, context)
            elif action == "customize_shortcuts":
                return await self._customize_shortcuts(params, context)
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

    async def _change_password(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Change password with context"""
        try:
            # Get security analysis from context
            analysis = context.get("security_analysis", {})
            recommendations = context.get("recommendations", {})
            
            # Apply password requirements
            requirements = self.get_config("settings.security.password_requirements", {})
            
            # Validate new password
            new_password = params.get("password_data", {}).get("new_password", "")
            strength = self._calculate_password_strength(new_password)
            
            if strength < 70:  # minimum acceptable strength
                return {
                    "success": False,
                    "message": "Password does not meet security requirements",
                    "strength": strength,
                    "recommendations": recommendations,
                    "context": context
                }
            
            return {
                "success": True,
                "message": "Password changed successfully",
                "strength": strength,
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
