from typing import Dict, Any, Optional
import json
from pathlib import Path
import tweepy
import schedule
import threading
import time
from .base_skill import BaseSkill
from .content_creation_skill import ContentCreationSkill
from .rag_skill import RAGSkill
from ..utils.validation import validate_params

class TwitterSkill(BaseSkill):
    """Twitter skill with content generation and RAG capabilities."""

    def __init__(self):
        super().__init__()
        self.name = "twitter"
        self.config = self._load_config()
        self.api = None
        self.credentials = None
        self.scheduler_thread = None
        self.scheduled_jobs = {}
        self.content_skill = ContentCreationSkill()
        self.rag_skill = RAGSkill()

    def _load_config(self) -> Dict[str, Any]:
        """Load Twitter skill configuration."""
        config_path = Path(__file__).parent / "configs" / "twitter.json"
        with open(config_path, "r") as f:
            return json.load(f)

    def set_credentials(self, credentials: Dict[str, str]) -> bool:
        """Set Twitter API credentials."""
        try:
            auth = tweepy.OAuthHandler(
                credentials["api_key"],
                credentials["api_secret"]
            )
            auth.set_access_token(
                credentials["access_token"],
                credentials["access_token_secret"]
            )
            self.api = tweepy.API(auth)
            self.api.verify_credentials()
            self.credentials = credentials
            return True
        except Exception as e:
            print(f"Error setting credentials: {str(e)}")
            return False

    async def _generate_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content using specified source."""
        content_source = params["content_source"]
        content_params = params.get("content_params", {})

        if content_source == "generated":
            # Use content creation skill
            result = await self.content_skill.execute({
                "action": "generate",
                "context": content_params.get("context", {}),
                "query": content_params.get("query", "")
            })
        elif content_source == "rag":
            # Use RAG skill
            if "data_path" in content_params:
                # Initialize RAG if data_path provided
                init_result = self.rag_skill.execute({
                    "action": "initialize",
                    "data_path": content_params["data_path"]
                })
                if not init_result["success"]:
                    return init_result

            result = self.rag_skill.execute({
                "action": "generate",
                "query": content_params.get("query", ""),
                "context": content_params.get("context", {})
            })
        else:
            return {
                "success": False,
                "errors": [f"Unknown content source: {content_source}"]
            }

        if not result["success"]:
            return result

        return {
            "success": True,
            "content": result.get("content", ""),
            "reasoning": result.get("reasoning", {})
        }

    def _post_tweet(self, content: str) -> Dict[str, Any]:
        """Post a tweet."""
        try:
            if not self.api:
                return {
                    "success": False,
                    "errors": ["Twitter API credentials not configured"]
                }

            tweet = self.api.update_status(content)
            return {
                "success": True,
                "tweet_id": tweet.id_str,
                "metrics": {
                    "likes": tweet.favorite_count,
                    "retweets": tweet.retweet_count
                }
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)]
            }

    def _schedule_post(self, content: str, schedule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a tweet."""
        try:
            job_id = f"tweet_{int(time.time())}"
            
            def scheduled_job():
                self._post_tweet(content)
            
            if schedule_config["frequency"] == "daily":
                schedule.every().day.at(schedule_config["time"]).do(scheduled_job)
            elif schedule_config["frequency"] == "weekly":
                schedule.every().week.at(schedule_config["time"]).do(scheduled_job)
            elif schedule_config["frequency"] == "monthly":
                schedule.every(30).days.at(schedule_config["time"]).do(scheduled_job)
            
            self.scheduled_jobs[job_id] = scheduled_job
            
            if not self.scheduler_thread or not self.scheduler_thread.is_alive():
                self.scheduler_thread = threading.Thread(target=self._run_scheduler)
                self.scheduler_thread.daemon = True
                self.scheduler_thread.start()
            
            return {
                "success": True,
                "job_id": job_id,
                "schedule": schedule_config
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)]
            }

    def _run_scheduler(self):
        """Run the scheduler thread."""
        while True:
            schedule.run_pending()
            time.sleep(60)

    def _monitor_interactions(self, tweet_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor tweet interactions."""
        try:
            tweet = self.api.get_status(tweet_id)
            metrics = {
                "likes": tweet.favorite_count,
                "retweets": tweet.retweet_count,
                "replies": len(list(tweepy.Cursor(self.api.search_tweets, 
                    q=f"to:{tweet.user.screen_name}", 
                    since_id=tweet_id).items()))
            }
            
            threshold_reached = any(
                metrics[metric] >= config["interaction_threshold"]
                for metric in config["track_metrics"]
                if metric in metrics
            )
            
            return {
                "success": True,
                "metrics": metrics,
                "threshold_reached": threshold_reached
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)]
            }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Twitter skill operations."""
        try:
            validate_params(params, self.config["params_schema"])
            action = params["action"]

            if action == "post":
                if "content" in params:
                    # Direct content posting
                    return self._post_tweet(params["content"])
                else:
                    # Generate and post content
                    content_result = await self._generate_content(params)
                    if not content_result["success"]:
                        return content_result
                    return self._post_tweet(content_result["content"])

            elif action == "schedule":
                content_result = await self._generate_content(params)
                if not content_result["success"]:
                    return content_result
                return self._schedule_post(content_result["content"], params["schedule"])

            elif action == "monitor":
                if "tweet_id" not in params or "monitoring" not in params:
                    return {
                        "success": False,
                        "errors": ["tweet_id and monitoring config required"]
                    }
                return self._monitor_interactions(params["tweet_id"], params["monitoring"])

            else:
                return {
                    "success": False,
                    "errors": [f"Unknown action: {action}"]
                }

        except Exception as e:
            return {
                "success": False,
                "errors": [str(e)]
            }

    def cleanup(self):
        """Clean up resources."""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            schedule.clear()
            self.scheduled_jobs = {}
