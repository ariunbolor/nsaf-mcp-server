from typing import Dict, Any, List
import openai
from .base_skill import BaseSkill

class ContentCreationSkill(BaseSkill):
    def __init__(self):
        super().__init__(
            name="content_creation",
            description="Create content for various platforms using AI"
        )
        self.required_credentials = ['openai_api_key']
        
        # Default system-level configurations
        self.default_system_config = {
            'config_schema': {
                'type': 'object',
                'properties': {
                    'model': {
                        'type': 'string',
                        'enum': ['gpt-4', 'gpt-3.5-turbo'],
                        'default': 'gpt-4'
                    },
                    'max_tokens': {
                        'type': 'integer',
                        'minimum': 1,
                        'maximum': 4000,
                        'default': 1000
                    },
                    'temperature': {
                        'type': 'number',
                        'minimum': 0,
                        'maximum': 2,
                        'default': 0.7
                    }
                }
            },
            'params_schema': {
                'type': 'object',
                'required': ['platform', 'content_type', 'topic'],
                'properties': {
                    'platform': {
                        'type': 'string',
                        'enum': ['facebook', 'instagram', 'twitter', 'linkedin']
                    },
                    'content_type': {
                        'type': 'string',
                        'enum': ['text', 'image_prompt', 'video_script']
                    },
                    'topic': {
                        'type': 'string'
                    },
                    'tone': {
                        'type': 'string',
                        'enum': ['professional', 'casual', 'humorous', 'formal'],
                        'default': 'professional'
                    },
                    'target_audience': {
                        'type': 'string'
                    },
                    'keywords': {
                        'type': 'array',
                        'items': {'type': 'string'}
                    }
                }
            }
        }
        
        # Platform-specific content guidelines
        self.platform_guidelines = {
            'facebook': {
                'text': {
                    'max_length': 63206,
                    'optimal_length': 100,
                    'format_tips': [
                        'Use emojis sparingly',
                        'Include a clear call-to-action',
                        'Keep paragraphs short'
                    ]
                }
            },
            'instagram': {
                'text': {
                    'max_length': 2200,
                    'optimal_length': 150,
                    'format_tips': [
                        'Use relevant hashtags',
                        'Include emojis',
                        'Space out paragraphs'
                    ]
                }
            },
            'twitter': {
                'text': {
                    'max_length': 280,
                    'optimal_length': 240,
                    'format_tips': [
                        'Use hashtags strategically',
                        'Include media when possible',
                        'End with a call-to-action'
                    ]
                }
            },
            'linkedin': {
                'text': {
                    'max_length': 3000,
                    'optimal_length': 1200,
                    'format_tips': [
                        'Use professional tone',
                        'Include industry insights',
                        'Format with bullet points'
                    ]
                }
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters for content creation"""
        errors = []
        
        # Check required parameters
        required = ['platform', 'content_type', 'topic']
        for param in required:
            if param not in params:
                errors.append(f"Missing required parameter: {param}")
        
        if not errors:
            # Validate platform
            if params['platform'] not in self.platform_guidelines:
                errors.append(f"Unsupported platform: {params['platform']}")
            
            # Validate content type
            valid_types = ['text', 'image_prompt', 'video_script']
            if params['content_type'] not in valid_types:
                errors.append(f"Unsupported content type: {params['content_type']}")
            
            # Validate topic
            if not params['topic'].strip():
                errors.append("Topic cannot be empty")
        
        return errors

    def execute(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create content based on parameters"""
        params = params or {}
        
        # Validate parameters
        errors = self.validate_params(params)
        if errors:
            return {
                'success': False,
                'errors': errors
            }
        
        # Check credentials
        missing_creds = self.check_credentials()
        if missing_creds:
            return {
                'success': False,
                'errors': [f"Missing credentials: {', '.join(missing_creds)}"]
            }
        
        try:
            # Configure OpenAI
            openai.api_key = self.get_config('credentials.openai_api_key')
            
            # Get platform guidelines
            platform = params['platform']
            content_type = params['content_type']
            guidelines = self.platform_guidelines[platform][content_type]
            
            # Prepare prompt
            prompt = self._generate_prompt(params, guidelines)
            
            # Generate content
            response = openai.ChatCompletion.create(
                model=self.get_config('model', 'gpt-4'),
                messages=[
                    {"role": "system", "content": prompt['system']},
                    {"role": "user", "content": prompt['user']}
                ],
                max_tokens=self.get_config('max_tokens', 1000),
                temperature=self.get_config('temperature', 0.7)
            )
            
            content = response.choices[0].message.content
            
            # Post-process content
            processed_content = self._post_process_content(content, platform, content_type)
            
            return {
                'success': True,
                'content': processed_content,
                'platform': platform,
                'type': content_type,
                'metadata': {
                    'length': len(processed_content),
                    'guidelines_followed': True,
                    'platform_specific_metrics': self._get_content_metrics(processed_content, platform, content_type)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [str(e)]
            }

    def _generate_prompt(self, params: Dict[str, Any], guidelines: Dict[str, Any]) -> Dict[str, str]:
        """Generate system and user prompts"""
        platform = params['platform']
        content_type = params['content_type']
        
        system_prompt = f"""You are a professional content creator for {platform}.
Your task is to create {content_type} content following these guidelines:
- Maximum length: {guidelines['max_length']} characters
- Optimal length: {guidelines['optimal_length']} characters
- Format tips: {', '.join(guidelines['format_tips'])}

Ensure the content is:
1. Engaging and platform-appropriate
2. Optimized for the platform's algorithm
3. Following brand voice and tone
4. Including relevant calls-to-action"""

        user_prompt = f"""Create {content_type} content for {platform} about: {params['topic']}
Tone: {params.get('tone', 'professional')}
Target Audience: {params.get('target_audience', 'general')}
Keywords: {', '.join(params.get('keywords', []))}"""

        return {
            'system': system_prompt,
            'user': user_prompt
        }

    def _post_process_content(self, content: str, platform: str, content_type: str) -> str:
        """Post-process generated content"""
        guidelines = self.platform_guidelines[platform][content_type]
        
        # Truncate if exceeds max length
        if len(content) > guidelines['max_length']:
            content = content[:guidelines['max_length']]
        
        # Platform-specific formatting
        if platform == 'instagram':
            # Add line breaks between paragraphs
            content = content.replace('\n', '\n\n')
        elif platform == 'twitter':
            # Ensure within character limit
            if len(content) > 280:
                content = content[:277] + "..."
        
        return content

    def _get_content_metrics(self, content: str, platform: str, content_type: str) -> Dict[str, Any]:
        """Calculate platform-specific metrics"""
        metrics = {
            'character_count': len(content),
            'word_count': len(content.split()),
            'url_count': content.count('http'),
            'hashtag_count': content.count('#'),
            'mention_count': content.count('@'),
            'emoji_count': sum(c.isemoji() for c in content) if hasattr(str, 'isemoji') else 0
        }
        
        # Platform-specific metrics
        if platform == 'twitter':
            metrics['remaining_characters'] = 280 - len(content)
        elif platform == 'instagram':
            metrics['paragraph_count'] = content.count('\n\n') + 1
        
        return metrics
