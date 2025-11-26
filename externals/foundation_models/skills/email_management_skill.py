from typing import Dict, Any, List, Optional
from .base_skill import BaseSkill
from .rag_skill import RAGSkill
from .content_creation_skill import ContentCreationSkill
import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time

class EmailManagementSkill(BaseSkill):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="email_management",
            description="A skill for managing email operations including checking, sending, organizing and managing emails",
            config=config
        )
        self.required_credentials = [
            "email_address",
            "email_password",
            "smtp_server",
            "imap_server"
        ]
        self.smtp = None
        self.imap = None
        self.rag_skill = RAGSkill()
        self.content_skill = ContentCreationSkill()
        self.metrics = {
            "emails_sent": 0,
            "emails_read": 0,
            "emails_deleted": 0,
            "emails_archived": 0,
            "emails_forwarded": 0,
            "attachments_sent": 0,
            "searches_performed": 0,
            "unsubscribes": 0,
            "response_rate": 0,
            "avg_response_time": 0,  # in minutes
            "successful_operations": 0,
            "failed_operations": 0,
            "total_processing_time": 0,  # in seconds
            "inbox_size": 0,
            "archive_size": 0,
            "last_check_time": None,
            "email_categories": {
                "urgent": 0,
                "important": 0,
                "normal": 0,
                "spam": 0
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters for email operations"""
        errors = []
        required_params = {
            "action": ["check", "send", "delete", "mark_read", "mark_unread", 
                      "forward", "archive", "attach", "search", "unsubscribe"],
            "email_id": ["delete", "mark_read", "mark_unread", "forward", "archive"],
            "recipient": ["send", "forward"],
            "subject": ["send"],
            "body": ["send"],
            "search_query": ["search"]
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
        """Execute email management operations with reasoning"""
        if params is None:
            params = {}

        # Validate parameters
        errors = self.validate_params(params)
        if errors:
            return {"success": False, "errors": errors}

        # Check credentials
        missing_creds = self.check_credentials()
        if missing_creds:
            return {
                "success": False,
                "errors": [f"Missing credentials: {', '.join(missing_creds)}"]
            }

        action = params["action"]
        try:
            # Connect to email servers if needed
            if not self._ensure_connections():
                return {
                    "success": False,
                    "errors": ["Failed to connect to email servers"]
                }

            # Apply reasoning based on action
            reasoning_result = await self._apply_reasoning(action, params)
            if not reasoning_result["success"]:
                return reasoning_result

            # Execute action with reasoning context
            result = await self._execute_action(action, params, reasoning_result.get("context", {}))
            
            # Update metrics
            self._update_metrics(action, result["success"])
            
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
        """Apply human-like reasoning to email operations"""
        try:
            reasoning_context = {
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "goal": self._determine_goal(action, params),
                "source": self._determine_source(params),
                "priority": self._determine_priority(params),
                "strategy": self._determine_strategy(action, params)
            }

            # For send action, determine if we need content generation
            if action == "send" and "body" not in params:
                content_result = await self._generate_content(params)
                if not content_result["success"]:
                    return content_result
                params["body"] = content_result["content"]
                reasoning_context["content_generation"] = content_result["reasoning"]

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
        """Determine the goal and sub-goals of the email operation"""
        goals = {
            "primary_goal": "",
            "sub_goals": [],
            "success_criteria": []
        }

        if action == "send":
            goals.update({
                "primary_goal": "Communicate information effectively and professionally",
                "sub_goals": [
                    "Ensure message clarity",
                    "Maintain professional tone",
                    "Include all necessary information",
                    "Verify recipient details"
                ],
                "success_criteria": [
                    "Message delivered successfully",
                    "Content is clear and complete",
                    "Professional formatting maintained",
                    "All attachments included"
                ]
            })
        elif action == "check":
            goals.update({
                "primary_goal": "Stay updated with incoming communications",
                "sub_goals": [
                    "Identify urgent messages",
                    "Categorize incoming emails",
                    "Update response metrics",
                    "Monitor inbox health"
                ],
                "success_criteria": [
                    "All new emails processed",
                    "Priority emails identified",
                    "Metrics updated accurately",
                    "Inbox organized effectively"
                ]
            })
        elif action == "search":
            goals.update({
                "primary_goal": "Find specific information in email history",
                "sub_goals": [
                    "Optimize search parameters",
                    "Filter relevant results",
                    "Consider time constraints",
                    "Track search patterns"
                ],
                "success_criteria": [
                    "Desired information found",
                    "Search completed efficiently",
                    "Results properly filtered",
                    "Search history updated"
                ]
            })
        elif action == "archive":
            goals.update({
                "primary_goal": "Maintain organized email storage",
                "sub_goals": [
                    "Categorize emails properly",
                    "Update archive structure",
                    "Maintain searchability",
                    "Track archive metrics"
                ],
                "success_criteria": [
                    "Emails properly archived",
                    "Categories maintained",
                    "Archive size optimized",
                    "Metrics updated"
                ]
            })
        elif action == "delete":
            goals.update({
                "primary_goal": "Remove unnecessary emails to maintain inbox cleanliness",
                "sub_goals": [
                    "Verify deletion criteria",
                    "Backup if needed",
                    "Update storage metrics",
                    "Maintain deletion log"
                ],
                "success_criteria": [
                    "Unnecessary emails removed",
                    "Important emails preserved",
                    "Storage optimized",
                    "Metrics updated"
                ]
            })
        else:
            goals.update({
                "primary_goal": "Manage email communications efficiently",
                "sub_goals": [
                    "Maintain organization",
                    "Ensure data integrity",
                    "Track operations",
                    "Update metrics"
                ],
                "success_criteria": [
                    "Operation completed successfully",
                    "System state maintained",
                    "Metrics updated",
                    "Integrity preserved"
                ]
            })

        return goals

    def _determine_source(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the source and context of the email content"""
        source_info = {
            "type": "direct_action",
            "context": {},
            "metadata": {},
            "requirements": []
        }

        if "body" in params:
            source_info.update({
                "type": "user_provided",
                "context": {
                    "content_length": len(params["body"]),
                    "has_formatting": bool(re.search(r'<[^>]+>', params["body"])),
                    "language": self._detect_language(params["body"])
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "contains_links": bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', params["body"]))
                },
                "requirements": [
                    "Preserve formatting" if source_info["context"]["has_formatting"] else "Plain text",
                    "URL validation" if source_info["metadata"]["contains_links"] else None
                ]
            })
        elif "content_source" in params:
            source_info.update({
                "type": params["content_source"],
                "context": {
                    "source_type": params.get("source_type", "unknown"),
                    "template_id": params.get("template_id"),
                    "version": params.get("version", "1.0")
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "template_name": params.get("template_name")
                },
                "requirements": [
                    "Template validation",
                    "Version check",
                    "Content generation"
                ]
            })

        return source_info

    def _determine_priority(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the priority and urgency of the email operation"""
        priority_info = {
            "level": "normal",
            "urgency": "low",
            "factors": [],
            "handling": {}
        }

        # Check explicit priority
        if "priority" in params:
            priority_info["level"] = params["priority"]
            priority_info["factors"].append("User specified priority")

        # Check subject for urgency keywords
        if "subject" in params:
            subject_lower = params["subject"].lower()
            urgency_keywords = {
                "high": ["urgent", "asap", "emergency", "critical"],
                "medium": ["important", "attention", "priority"],
                "low": ["fyi", "update", "newsletter"]
            }

            for level, keywords in urgency_keywords.items():
                if any(kw in subject_lower for kw in keywords):
                    priority_info["urgency"] = level
                    priority_info["factors"].append(f"Subject indicates {level} urgency")
                    break

        # Check timing factors
        current_hour = datetime.now().hour
        if current_hour < 9 or current_hour > 17:  # Outside business hours
            priority_info["factors"].append("Outside business hours")
            if priority_info["urgency"] == "high":
                priority_info["level"] = "high"

        # Set handling recommendations
        priority_info["handling"] = {
            "response_time": "immediate" if priority_info["urgency"] == "high" else 
                           "same_day" if priority_info["urgency"] == "medium" else 
                           "within_48_hours",
            "notification": priority_info["level"] == "high",
            "escalation": priority_info["urgency"] == "high" and priority_info["level"] == "high",
            "tracking": True
        }

        return priority_info

    def _determine_strategy(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive strategy for handling the email operation"""
        strategy = {
            "approach": "",
            "steps": [],
            "validations": [],
            "fallback": {},
            "optimization": {}
        }

        if action == "send":
            strategy.update({
                "approach": "Ensure clear communication and proper formatting",
                "steps": [
                    "Validate recipient addresses",
                    "Check content formatting",
                    "Verify attachments",
                    "Apply email signature",
                    "Set appropriate headers"
                ],
                "validations": [
                    "Email address format",
                    "Content completeness",
                    "Attachment size limits",
                    "Spam trigger words"
                ],
                "fallback": {
                    "delivery_failure": "Retry with alternate server",
                    "attachment_error": "Split into multiple emails",
                    "format_issue": "Convert to plain text"
                },
                "optimization": {
                    "batch_processing": False,
                    "compression": "if_needed",
                    "scheduling": "immediate"
                }
            })
        elif action == "check":
            strategy.update({
                "approach": "Efficiently process and organize incoming emails",
                "steps": [
                    "Connect to IMAP server",
                    "Fetch new messages",
                    "Apply filters",
                    "Update metrics",
                    "Organize inbox"
                ],
                "validations": [
                    "Server connection",
                    "Message integrity",
                    "Filter rules",
                    "Storage space"
                ],
                "fallback": {
                    "connection_error": "Retry with backoff",
                    "filter_error": "Skip to manual sort",
                    "space_issue": "Archive old messages"
                },
                "optimization": {
                    "batch_processing": True,
                    "caching": "enabled",
                    "parallel_processing": "if_supported"
                }
            })
        elif action == "search":
            strategy.update({
                "approach": "Use optimal search parameters for accurate results",
                "steps": [
                    "Parse search query",
                    "Apply search filters",
                    "Sort results",
                    "Cache results",
                    "Update search history"
                ],
                "validations": [
                    "Query syntax",
                    "Filter validity",
                    "Result limit",
                    "Cache size"
                ],
                "fallback": {
                    "no_results": "Broaden search",
                    "timeout": "Reduce scope",
                    "syntax_error": "Use simple search"
                },
                "optimization": {
                    "index_usage": True,
                    "result_caching": True,
                    "parallel_search": "if_available"
                }
            })

        return strategy

    async def _generate_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate email content using RAG or AI generation"""
        try:
            content_source = params.get("content_source", "generated")
            content_params = params.get("content_params", {})

            if content_source == "rag":
                result = self.rag_skill.execute({
                    "action": "generate",
                    "query": content_params.get("query", ""),
                    "context": content_params
                })
            else:
                result = await self.content_skill.execute({
                    "action": "generate",
                    "context": content_params,
                    "style": "professional_email"
                })

            if not result["success"]:
                return result

            return {
                "success": True,
                "content": result["content"],
                "reasoning": result.get("reasoning", {})
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"Content generation error: {str(e)}"]
            }

    def _ensure_connections(self) -> bool:
        """Ensure connections to email servers are established"""
        try:
            if not self.smtp:
                self.smtp = smtplib.SMTP(self.get_config("credentials.smtp_server"))
                self.smtp.starttls()
                self.smtp.login(
                    self.get_config("credentials.email_address"),
                    self.get_config("credentials.email_password")
                )

            if not self.imap:
                self.imap = imaplib.IMAP4_SSL(self.get_config("credentials.imap_server"))
                self.imap.login(
                    self.get_config("credentials.email_address"),
                    self.get_config("credentials.email_password")
                )

            return True
        except Exception as e:
            print(f"Connection error: {str(e)}")
            return False

    def _update_metrics(self, action: str, success: bool, params: Dict[str, Any] = None) -> None:
        """Update comprehensive skill metrics based on action and result"""
        # Update basic operation metrics
        if success:
            self.metrics["successful_operations"] += 1
            
            # Update action-specific metrics
            if action == "send":
                self.metrics["emails_sent"] += 1
                if params and "attachment_path" in params:
                    self.metrics["attachments_sent"] += 1
            elif action == "check":
                self.metrics["emails_read"] += 1
                self.metrics["last_check_time"] = datetime.now().isoformat()
            elif action == "delete":
                self.metrics["emails_deleted"] += 1
            elif action == "archive":
                self.metrics["emails_archived"] += 1
            elif action == "forward":
                self.metrics["emails_forwarded"] += 1
            elif action == "search":
                self.metrics["searches_performed"] += 1
            elif action == "unsubscribe":
                self.metrics["unsubscribes"] += 1

            # Update timing metrics
            if params and "response_to" in params:
                response_time = (datetime.now() - datetime.fromisoformat(params["response_to"])).total_seconds() / 60
                if self.metrics["avg_response_time"] == 0:
                    self.metrics["avg_response_time"] = response_time
                else:
                    self.metrics["avg_response_time"] = (
                        self.metrics["avg_response_time"] * 0.8 + response_time * 0.2
                    )

            # Update email categorization
            if params and "category" in params:
                self.metrics["email_categories"][params["category"]] += 1

            # Update storage metrics
            if action in ["check", "archive", "delete"]:
                self._update_storage_metrics()

        else:
            self.metrics["failed_operations"] += 1

    def _update_storage_metrics(self) -> None:
        """Update email storage metrics"""
        try:
            # Get inbox size
            self.imap.select('INBOX')
            _, messages = self.imap.search(None, 'ALL')
            self.metrics["inbox_size"] = len(messages[0].split())

            # Get archive size
            archive_folder = self.get_config("settings.archive_folder", "Archive")
            self.imap.select(archive_folder)
            _, archived = self.imap.search(None, 'ALL')
            self.metrics["archive_size"] = len(archived[0].split())

        except Exception as e:
            print(f"Error updating storage metrics: {str(e)}")

    def _detect_language(self, text: str) -> str:
        """Detect the language of the text"""
        # Simple language detection based on common words
        # In practice, you might want to use a proper language detection library
        english_words = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that'])
        spanish_words = set(['el', 'la', 'de', 'que', 'y', 'en', 'un', 'ser'])
        french_words = set(['le', 'la', 'de', 'et', 'en', 'un', 'être', 'avoir'])

        words = set(text.lower().split())
        
        en_count = len(words.intersection(english_words))
        es_count = len(words.intersection(spanish_words))
        fr_count = len(words.intersection(french_words))
        
        counts = {'en': en_count, 'es': es_count, 'fr': fr_count}
        max_lang = max(counts.items(), key=lambda x: x[1])
        
        return max_lang[0] if max_lang[1] > 0 else 'unknown'

    async def _execute_action(self, action: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute email action with context"""
        try:
            if action == "check":
                return await self._check_emails(context)
            elif action == "send":
                return await self._send_email(params, context)
            elif action == "delete":
                return await self._delete_email(params, context)
            elif action == "mark_read":
                return await self._mark_email(params, True, context)
            elif action == "mark_unread":
                return await self._mark_email(params, False, context)
            elif action == "forward":
                return await self._forward_email(params, context)
            elif action == "archive":
                return await self._archive_email(params, context)
            elif action == "attach":
                return await self._attach_file(params, context)
            elif action == "search":
                return await self._search_emails(params, context)
            elif action == "unsubscribe":
                return await self._unsubscribe(params, context)
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

    async def _check_emails(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check emails in inbox with context"""
        try:
            self.imap.select('INBOX')
            _, messages = self.imap.search(None, 'ALL')
            
            emails = []
            for num in messages[0].split():
                _, msg = self.imap.fetch(num, '(RFC822)')
                email_body = msg[0][1]
                email_message = email.message_from_bytes(email_body)
                emails.append({
                    'subject': email_message['subject'],
                    'from': email_message['from'],
                    'date': email_message['date']
                })

            return {
                "success": True,
                "message": "Emails retrieved successfully",
                "emails": emails,
                "context": context
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "context": context
            }

    async def _send_email(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Send an email with context"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.get_config("credentials.email_address")
            msg['To'] = params['recipient']
            msg['Subject'] = params['subject']
            msg.attach(MIMEText(params['body'], 'plain'))

            self.smtp.send_message(msg)
            
            return {
                "success": True,
                "message": "Email sent successfully",
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
        if self.smtp:
            try:
                self.smtp.quit()
            except:
                pass
        if self.imap:
            try:
                self.imap.logout()
            except:
                pass
