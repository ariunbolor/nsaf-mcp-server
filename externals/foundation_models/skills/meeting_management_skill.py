from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pytz
from .base_skill import BaseSkill
from .rag_skill import RAGSkill
from .content_creation_skill import ContentCreationSkill

class MeetingManagementSkill(BaseSkill):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="meeting_management",
            description="A skill for managing meetings including scheduling, organizing, and handling meeting-related tasks",
            config=config
        )
        self.required_credentials = [
            "calendar_api_key",
            "video_conference_api_key"
        ]
        self.rag_skill = RAGSkill()
        self.content_skill = ContentCreationSkill()
        self.metrics = {
            "meetings_scheduled": 0,
            "meetings_completed": 0,
            "meetings_cancelled": 0,
            "meetings_rescheduled": 0,
            "total_participants": 0,
            "unique_participants": set(),  # track unique participants
            "avg_attendance_rate": 0,
            "avg_meeting_duration": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_meeting_minutes": 0,
            "recordings_created": 0,
            "transcripts_generated": 0,
            "notes_taken": 0,
            "video_conferences": 0,
            "screen_shares": 0,
            "timezone_conflicts_resolved": 0,
            "meeting_categories": {
                "team_sync": 0,
                "client_meeting": 0,
                "interview": 0,
                "workshop": 0,
                "other": 0
            },
            "time_distribution": {
                "morning": 0,    # 6-12
                "afternoon": 0,  # 12-17
                "evening": 0     # 17-22
            },
            "participant_engagement": {
                "camera_on_rate": 0,
                "speaking_time_distribution": 0,  # standard deviation
                "interaction_score": 0  # 0-100
            },
            "technical_metrics": {
                "av_issues": 0,
                "connection_drops": 0,
                "successful_recordings": 0,
                "failed_recordings": 0
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters for meeting operations"""
        errors = []
        required_params = {
            "action": ["schedule", "invite", "check_timezone", "take_notes", 
                      "setup_video", "check_av", "share_screen", "upload_recording",
                      "download_transcript", "reschedule", "cancel"],
            "meeting_id": ["invite", "take_notes", "share_screen", "upload_recording",
                         "download_transcript", "reschedule", "cancel"],
            "datetime": ["schedule", "reschedule"],
            "duration": ["schedule"],
            "participants": ["schedule", "invite"],
            "timezone": ["schedule", "check_timezone"],
            "title": ["schedule"],
            "notes": ["take_notes"],
            "recording_path": ["upload_recording"]
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
        """Execute meeting management operations with reasoning"""
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
        """Apply human-like reasoning to meeting operations"""
        try:
            reasoning_context = {
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "goal": self._determine_goal(action, params),
                "priority": self._determine_priority(params),
                "strategy": self._determine_strategy(action, params),
                "considerations": self._determine_considerations(action, params)
            }

            # For note-taking, determine if we need content generation
            if action == "take_notes" and "notes" not in params:
                content_result = await self._generate_content(params)
                if not content_result["success"]:
                    return content_result
                params["notes"] = content_result["content"]
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
        """Determine the comprehensive goal of the meeting operation"""
        goals = {
            "primary_goal": "",
            "sub_goals": [],
            "success_criteria": [],
            "constraints": [],
            "optimization_targets": []
        }

        if action == "schedule":
            goals.update({
                "primary_goal": "Organize an effective meeting at a suitable time for all participants",
                "sub_goals": [
                    "Find optimal time slot",
                    "Ensure participant availability",
                    "Set up necessary resources",
                    "Send clear communications"
                ],
                "success_criteria": [
                    "All key participants can attend",
                    "No timezone conflicts",
                    "Resources available",
                    "Clear agenda set"
                ],
                "constraints": [
                    "Working hours in all timezones",
                    "Participant calendar conflicts",
                    "Resource availability",
                    "Lead time requirements"
                ],
                "optimization_targets": [
                    "Minimize timezone spread",
                    "Maximize attendance",
                    "Optimal duration",
                    "Resource efficiency"
                ]
            })
        elif action == "take_notes":
            goals.update({
                "primary_goal": "Document important meeting discussions and decisions",
                "sub_goals": [
                    "Capture key points",
                    "Record action items",
                    "Note decisions made",
                    "Track follow-ups"
                ],
                "success_criteria": [
                    "All key points documented",
                    "Action items assigned",
                    "Decisions clearly recorded",
                    "Follow-ups scheduled"
                ],
                "constraints": [
                    "Note-taking permissions",
                    "Confidentiality requirements",
                    "Format restrictions",
                    "Distribution limits"
                ],
                "optimization_targets": [
                    "Clarity of documentation",
                    "Actionability of items",
                    "Accessibility of notes",
                    "Follow-up tracking"
                ]
            })
        elif action == "setup_video":
            goals.update({
                "primary_goal": "Ensure smooth video conferencing experience",
                "sub_goals": [
                    "Test connection quality",
                    "Configure audio/video",
                    "Set up sharing features",
                    "Prepare backup options"
                ],
                "success_criteria": [
                    "Stable connection established",
                    "Clear audio/video quality",
                    "Features functioning",
                    "Backups ready"
                ],
                "constraints": [
                    "Bandwidth limitations",
                    "Platform restrictions",
                    "Security requirements",
                    "User permissions"
                ],
                "optimization_targets": [
                    "Connection stability",
                    "Audio/video quality",
                    "Feature availability",
                    "User experience"
                ]
            })
        else:
            goals.update({
                "primary_goal": "Manage meeting effectively and professionally",
                "sub_goals": [
                    "Ensure smooth execution",
                    "Maintain professionalism",
                    "Track outcomes",
                    "Handle issues"
                ],
                "success_criteria": [
                    "Meeting objectives met",
                    "Professional standards maintained",
                    "Issues resolved promptly",
                    "Outcomes documented"
                ],
                "constraints": [
                    "Time limitations",
                    "Resource availability",
                    "Technical constraints",
                    "Policy requirements"
                ],
                "optimization_targets": [
                    "Meeting effectiveness",
                    "Resource utilization",
                    "Participant satisfaction",
                    "Issue resolution"
                ]
            })

        return goals

    def _determine_priority(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive priority of the meeting operation"""
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

        # Check title for urgency keywords
        if "title" in params:
            title_lower = params["title"].lower()
            urgency_keywords = {
                "high": ["urgent", "asap", "emergency", "critical"],
                "medium": ["important", "priority", "review"],
                "low": ["routine", "regular", "weekly"]
            }

            for level, keywords in urgency_keywords.items():
                if any(kw in title_lower for kw in keywords):
                    priority_info["urgency"] = level
                    priority_info["factors"].append(f"Title indicates {level} urgency")
                    break

        # Check participant roles if available
        if "participant_roles" in params:
            roles = params["participant_roles"]
            if any(role in ["executive", "client", "external"] for role in roles):
                priority_info["level"] = "high"
                priority_info["factors"].append("Key stakeholders involved")

        # Check timing factors
        if "datetime" in params:
            meeting_time = datetime.fromisoformat(params["datetime"])
            lead_time = (meeting_time - datetime.now()).days
            
            if lead_time < 1:
                priority_info["urgency"] = "high"
                priority_info["factors"].append("Short lead time")
            elif lead_time < 3:
                priority_info["urgency"] = "medium"
                priority_info["factors"].append("Moderate lead time")

        # Determine impact
        priority_info["impact"] = {
            "schedule_impact": "high" if priority_info["urgency"] == "high" else "normal",
            "resource_needs": "priority" if priority_info["level"] == "high" else "standard",
            "notification_level": "immediate" if priority_info["urgency"] == "high" else "standard"
        }

        # Set handling recommendations
        priority_info["handling"] = {
            "scheduling_window": "same_day" if priority_info["urgency"] == "high" else 
                               "this_week" if priority_info["urgency"] == "medium" else 
                               "next_available",
            "notifications": priority_info["level"] == "high",
            "reminders": True if priority_info["level"] in ["high", "medium"] else False,
            "backup_plans": priority_info["level"] == "high"
        }

        return priority_info

    def _determine_strategy(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the comprehensive strategy for handling the meeting operation"""
        strategy = {
            "approach": "",
            "steps": [],
            "validations": [],
            "fallback": {},
            "optimization": {},
            "monitoring": {}
        }

        if action == "schedule":
            strategy.update({
                "approach": "Find optimal time considering all participants' availability",
                "steps": [
                    "Check participant calendars",
                    "Analyze timezone overlaps",
                    "Verify resource availability",
                    "Send calendar invites",
                    "Set up conferencing"
                ],
                "validations": [
                    "Participant responses",
                    "Room availability",
                    "Technical requirements",
                    "Calendar conflicts"
                ],
                "fallback": {
                    "time_conflict": "Suggest alternate slots",
                    "resource_unavailable": "Book alternative resource",
                    "timezone_issue": "Split into multiple sessions"
                },
                "optimization": {
                    "minimize_timezone_spread": True,
                    "prefer_core_hours": True,
                    "balance_participant_preferences": True
                },
                "monitoring": {
                    "track_responses": True,
                    "check_conflicts": True,
                    "verify_resources": True
                }
            })
        elif action == "take_notes":
            strategy.update({
                "approach": "Capture key points and action items clearly",
                "steps": [
                    "Prepare note template",
                    "Record key discussions",
                    "Track action items",
                    "Highlight decisions",
                    "Distribute notes"
                ],
                "validations": [
                    "Content completeness",
                    "Action item clarity",
                    "Decision documentation",
                    "Distribution list"
                ],
                "fallback": {
                    "missing_info": "Follow up with participants",
                    "unclear_decision": "Seek clarification",
                    "distribution_fail": "Use alternative channel"
                },
                "optimization": {
                    "use_templates": True,
                    "auto_categorize": True,
                    "tag_action_items": True
                },
                "monitoring": {
                    "track_completion": True,
                    "verify_accuracy": True,
                    "check_distribution": True
                }
            })
        elif action == "setup_video":
            strategy.update({
                "approach": "Ensure all technical requirements are met",
                "steps": [
                    "Test connection",
                    "Configure audio/video",
                    "Set up sharing",
                    "Prepare recording",
                    "Test backup options"
                ],
                "validations": [
                    "Connection speed",
                    "Audio quality",
                    "Video quality",
                    "Feature access"
                ],
                "fallback": {
                    "connection_issue": "Switch to backup connection",
                    "audio_problem": "Use phone audio",
                    "video_fail": "Disable video",
                    "sharing_issue": "Use alternative method"
                },
                "optimization": {
                    "optimize_bandwidth": True,
                    "auto_adjust_quality": True,
                    "preload_content": True
                },
                "monitoring": {
                    "track_performance": True,
                    "monitor_quality": True,
                    "log_issues": True
                }
            })

        return strategy

    def _determine_considerations(self, action: str, params: Dict[str, Any]) -> List[str]:
        """Determine important considerations for the meeting operation"""
        considerations = []
        
        if action == "schedule":
            considerations.extend([
                "Participant availability",
                "Time zone differences",
                "Meeting duration",
                "Technical requirements"
            ])
        elif action == "take_notes":
            considerations.extend([
                "Key discussion points",
                "Action items",
                "Decisions made",
                "Follow-up tasks"
            ])
        elif action == "setup_video":
            considerations.extend([
                "Internet connectivity",
                "Audio quality",
                "Video quality",
                "Screen sharing capability"
            ])
            
        return considerations

    async def _generate_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate meeting content using RAG or AI generation"""
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
                    "style": "meeting_notes"
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

    def _update_metrics(self, action: str, success: bool, params: Dict[str, Any]) -> None:
        """Update comprehensive skill metrics based on action and result"""
        if success:
            self.metrics["successful_operations"] += 1
            
            # Update action-specific metrics
            if action == "schedule":
                self.metrics["meetings_scheduled"] += 1
                if "participants" in params:
                    participants = params["participants"]
                    self.metrics["total_participants"] += len(participants)
                    self.metrics["unique_participants"].update(participants)
                if "duration" in params:
                    duration = params["duration"]
                    self.metrics["total_meeting_minutes"] += duration
                    self.metrics["avg_meeting_duration"] = (
                        self.metrics["total_meeting_minutes"] / self.metrics["meetings_scheduled"]
                    )
                # Update meeting category
                if "category" in params:
                    category = params["category"]
                    if category in self.metrics["meeting_categories"]:
                        self.metrics["meeting_categories"][category] += 1
                    else:
                        self.metrics["meeting_categories"]["other"] += 1
                
                # Update time distribution
                if "datetime" in params:
                    meeting_time = datetime.fromisoformat(params["datetime"])
                    hour = meeting_time.hour
                    if 6 <= hour < 12:
                        self.metrics["time_distribution"]["morning"] += 1
                    elif 12 <= hour < 17:
                        self.metrics["time_distribution"]["afternoon"] += 1
                    elif 17 <= hour < 22:
                        self.metrics["time_distribution"]["evening"] += 1

            elif action == "setup_video":
                self.metrics["video_conferences"] += 1
            elif action == "share_screen":
                self.metrics["screen_shares"] += 1
            elif action == "take_notes":
                self.metrics["notes_taken"] += 1
            elif action == "upload_recording":
                self.metrics["recordings_created"] += 1
                self.metrics["technical_metrics"]["successful_recordings"] += 1
            elif action == "download_transcript":
                self.metrics["transcripts_generated"] += 1
            elif action == "check_timezone":
                if params.get("conflicts_resolved", False):
                    self.metrics["timezone_conflicts_resolved"] += 1
            elif action == "reschedule":
                self.metrics["meetings_rescheduled"] += 1
            elif action == "cancel":
                self.metrics["meetings_cancelled"] += 1

            # Update engagement metrics if provided
            if "engagement_data" in params:
                data = params["engagement_data"]
                if "camera_on_percentage" in data:
                    self.metrics["participant_engagement"]["camera_on_rate"] = (
                        self.metrics["participant_engagement"]["camera_on_rate"] * 0.8 +
                        data["camera_on_percentage"] * 0.2
                    )
                if "speaking_time_std" in data:
                    self.metrics["participant_engagement"]["speaking_time_distribution"] = (
                        self.metrics["participant_engagement"]["speaking_time_distribution"] * 0.8 +
                        data["speaking_time_std"] * 0.2
                    )
                if "interaction_score" in data:
                    self.metrics["participant_engagement"]["interaction_score"] = (
                        self.metrics["participant_engagement"]["interaction_score"] * 0.8 +
                        data["interaction_score"] * 0.2
                    )

            # Update technical metrics if issues occurred
            if "technical_issues" in params:
                issues = params["technical_issues"]
                if "av_problems" in issues:
                    self.metrics["technical_metrics"]["av_issues"] += issues["av_problems"]
                if "connection_losses" in issues:
                    self.metrics["technical_metrics"]["connection_drops"] += issues["connection_losses"]

        else:
            self.metrics["failed_operations"] += 1
            if action == "upload_recording":
                self.metrics["technical_metrics"]["failed_recordings"] += 1

    async def _execute_action(self, action: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute meeting action with context"""
        try:
            if action == "schedule":
                return await self._schedule_meeting(params, context)
            elif action == "invite":
                return await self._send_invites(params, context)
            elif action == "check_timezone":
                return await self._check_timezone(params, context)
            elif action == "take_notes":
                return await self._take_notes(params, context)
            elif action == "setup_video":
                return await self._setup_video_conference(context)
            elif action == "check_av":
                return await self._check_av_equipment(context)
            elif action == "share_screen":
                return await self._share_screen(params, context)
            elif action == "upload_recording":
                return await self._upload_recording(params, context)
            elif action == "download_transcript":
                return await self._download_transcript(params, context)
            elif action == "reschedule":
                return await self._reschedule_meeting(params, context)
            elif action == "cancel":
                return await self._cancel_meeting(params, context)
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

    async def _schedule_meeting(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a new meeting with context"""
        try:
            # Implementation would create calendar event and set up video conference
            # For now, returning mock success
            return {
                "success": True,
                "message": "Meeting scheduled successfully",
                "context": context
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "context": context
            }

    async def _send_invites(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Send meeting invites with context"""
        try:
            # Implementation would send calendar invites
            # For now, returning mock success
            return {
                "success": True,
                "message": "Meeting invites sent successfully",
                "context": context
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)],
                "context": context
            }

    async def _check_timezone(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Check timezone compatibility with context"""
        try:
            timezone = params.get("timezone", "UTC")
            meeting_time = datetime.fromisoformat(params["datetime"])
            
            # Convert to target timezone
            target_tz = pytz.timezone(timezone)
            local_time = meeting_time.astimezone(target_tz)
            
            # Check if meeting time is during reasonable hours (8 AM - 6 PM)
            hour = local_time.hour
            if hour < 8 or hour > 18:
                return {
                    "success": False,
                    "warning": "Meeting time may be outside of business hours in target timezone",
                    "local_time": local_time.isoformat(),
                    "context": context
                }
            
            return {
                "success": True,
                "message": "Timezone check completed",
                "local_time": local_time.isoformat(),
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
