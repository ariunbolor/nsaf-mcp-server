from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from decimal import Decimal
from .base_skill import BaseSkill
from .rag_skill import RAGSkill

class OnlineShoppingSkill(BaseSkill):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="online_shopping",
            description="A skill for managing online shopping operations including cart management, payments, and order tracking",
            config=config
        )
        self.required_credentials = [
            "payment_api_key",  # For payment processing
            "shopping_api_key"   # For shopping platform integration
        ]
        self.rag_skill = RAGSkill()
        self.metrics = {
            "orders_placed": 0,
            "total_spent": Decimal('0.00'),
            "money_saved": Decimal('0.00'),  # through discounts
            "cart_items": 0,
            "successful_payments": 0,
            "failed_payments": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_order_value": Decimal('0.00'),
            "discounts_applied": 0,
            "reviews_analyzed": 0
        }

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters for shopping operations"""
        errors = []
        required_params = {
            "action": ["add_to_cart", "enter_payment", "check_order", "compare_prices",
                      "apply_discount", "download_invoice", "update_payment",
                      "manage_subscription", "read_reviews", "track_delivery"],
            "product_id": ["add_to_cart"],
            "payment_details": ["enter_payment", "update_payment"],
            "order_id": ["check_order", "download_invoice", "track_delivery"],
            "discount_code": ["apply_discount"],
            "subscription_id": ["manage_subscription"]
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
        """Execute shopping operations with reasoning"""
        if params is None:
            params = {}

        # Validate parameters
        errors = self.validate_params(params)
        if errors:
            return {"success": False, "errors": errors}

        # For payment operations, check credentials
        if params.get("action") in ["enter_payment", "update_payment"]:
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
        """Apply human-like reasoning to shopping operations"""
        try:
            reasoning_context = {
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "goal": self._determine_goal(action, params),
                "priority": self._determine_priority(params),
                "strategy": self._determine_strategy(action, params),
                "considerations": self._determine_considerations(action, params)
            }

            # For price comparison and reviews, get analysis
            if action in ["compare_prices", "read_reviews"]:
                analysis_result = await self._get_shopping_analysis(action, params)
                if not analysis_result["success"]:
                    return analysis_result
                reasoning_context["analysis"] = analysis_result["analysis"]
                reasoning_context["recommendations"] = analysis_result["recommendations"]

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
        """Determine the goal of the shopping operation"""
        if action == "compare_prices":
            return "Find the best value while considering price and quality"
        elif action == "enter_payment":
            return "Complete transaction securely and efficiently"
        elif action == "apply_discount":
            return "Maximize savings while ensuring discount validity"
        elif action == "read_reviews":
            return "Gather reliable product feedback for informed decision"
        return "Complete shopping operation while optimizing value and experience"

    def _determine_priority(self, params: Dict[str, Any]) -> str:
        """Determine the priority of the shopping operation"""
        if "priority" in params:
            return params["priority"]
        
        # Check operation type and cart value for priority
        if params.get("action") in ["enter_payment", "track_delivery"]:
            return "high"
        return "normal"

    def _determine_strategy(self, action: str, params: Dict[str, Any]) -> str:
        """Determine the strategy for handling the shopping operation"""
        if action == "compare_prices":
            return "Analyze prices across platforms considering shipping and taxes"
        elif action == "enter_payment":
            return "Process payment securely with proper validation"
        elif action == "read_reviews":
            return "Focus on verified purchases and detailed feedback"
        return "Execute operation while ensuring optimal user experience"

    def _determine_considerations(self, action: str, params: Dict[str, Any]) -> List[str]:
        """Determine important considerations for the shopping operation"""
        considerations = []
        
        if action == "compare_prices":
            considerations.extend([
                "Base price",
                "Shipping costs",
                "Delivery time",
                "Seller reputation"
            ])
        elif action == "enter_payment":
            considerations.extend([
                "Payment security",
                "Transaction fees",
                "Processing time",
                "Refund policy"
            ])
        elif action == "read_reviews":
            considerations.extend([
                "Review authenticity",
                "Rating distribution",
                "Common issues",
                "Recent feedback"
            ])
            
        return considerations

    async def _get_shopping_analysis(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get shopping analysis using RAG"""
        try:
            result = self.rag_skill.execute({
                "action": "generate",
                "query": f"{action} analysis",
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
                "errors": [f"Analysis error: {str(e)}"]
            }

    def _update_metrics(self, action: str, success: bool, params: Dict[str, Any]) -> None:
        """Update skill metrics based on action and result"""
        if success:
            self.metrics["successful_operations"] += 1
            
            if action == "add_to_cart":
                self.metrics["cart_items"] += 1
            elif action == "enter_payment":
                if params.get("amount"):
                    amount = Decimal(str(params["amount"]))
                    self.metrics["total_spent"] += amount
                    self.metrics["successful_payments"] += 1
                    # Update average order value
                    self.metrics["avg_order_value"] = (
                        self.metrics["total_spent"] / self.metrics["successful_payments"]
                    )
            elif action == "apply_discount":
                if params.get("discount_amount"):
                    self.metrics["money_saved"] += Decimal(str(params["discount_amount"]))
                    self.metrics["discounts_applied"] += 1
            elif action == "read_reviews":
                self.metrics["reviews_analyzed"] += 1
        else:
            self.metrics["failed_operations"] += 1
            if action == "enter_payment":
                self.metrics["failed_payments"] += 1

    async def _execute_action(self, action: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shopping action with context"""
        try:
            if action == "add_to_cart":
                return await self._add_to_cart(params, context)
            elif action == "enter_payment":
                return await self._enter_payment(params, context)
            elif action == "check_order":
                return await self._check_order(params, context)
            elif action == "compare_prices":
                return await self._compare_prices(params, context)
            elif action == "apply_discount":
                return await self._apply_discount(params, context)
            elif action == "download_invoice":
                return await self._download_invoice(params, context)
            elif action == "update_payment":
                return await self._update_payment(params, context)
            elif action == "manage_subscription":
                return await self._manage_subscription(params, context)
            elif action == "read_reviews":
                return await self._read_reviews(params, context)
            elif action == "track_delivery":
                return await self._track_delivery(params, context)
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

    async def _compare_prices(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare prices with context"""
        try:
            # Get price analysis from context
            analysis = context.get("analysis", {})
            recommendations = context.get("recommendations", {})
            
            # Apply price comparison strategy
            strategy = context.get("strategy", "")
            considerations = context.get("considerations", [])
            
            comparison_result = {
                "prices": [],  # Would contain actual price data
                "best_value": recommendations.get("best_value"),
                "price_range": recommendations.get("price_range"),
                "factors_considered": considerations
            }
            
            return {
                "success": True,
                "message": "Price comparison completed",
                "comparison": comparison_result,
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
