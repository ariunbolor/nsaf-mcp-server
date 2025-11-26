"""
Agent Creation Handler
--------------------
Handles the chat-based agent creation process.
"""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from ..skills.registry import SkillRegistry
from ..core.agent_builder import AgentBuilder
from ..utils.validation import validate_skill_config
from ..core.reasoning_engine import ReasoningEngine
from .websocket_handler import websocket_manager
from .quantum_symbolic_handler import QuantumSymbolicHandler

class AgentCreationHandler:
    def __init__(self):
        self.skill_registry = SkillRegistry()
        self.agent_builder = AgentBuilder()
        self.reasoning_engine = ReasoningEngine()
        self.quantum_handler = QuantumSymbolicHandler()
        self.websocket_manager = websocket_manager
        self.pending_skills = {}  # Store skills pending admin approval

    def _determine_capabilities(self, domain: Dict[str, Any], objectives: List[str]) -> List[str]:
        """Determine required capabilities based on domain and objectives"""
        capabilities = []
        
        # Add domain-specific capabilities
        if domain['primary'] == 'marketing':
            capabilities.extend(['content_analysis', 'engagement_monitoring'])
        elif domain['primary'] == 'support':
            capabilities.extend(['query_understanding', 'response_generation'])
        elif domain['primary'] == 'analytics':
            capabilities.extend(['data_analysis', 'pattern_recognition'])
            
        # Add objective-specific capabilities
        if 'brand_awareness' in objectives:
            capabilities.extend(['content_creation', 'social_media_management'])
        if 'lead_generation' in objectives:
            capabilities.extend(['lead_scoring', 'automation'])
        if 'engagement' in objectives:
            capabilities.extend(['conversation_management', 'sentiment_analysis'])
            
        return list(set(capabilities))  # Remove duplicates

    def _analyze_quantum_benefits(self, capabilities: List[str]) -> Dict[str, Any]:
        """Analyze if quantum symbolic computation would be beneficial"""
        quantum_relevant_capabilities = {
            'pattern_recognition': {
                'benefit': 'Enhanced pattern matching using quantum circuits',
                'config': {
                    'computationType': 'PATTERN_MATCHING',
                    'numQubits': 5,
                    'circuitDepth': 3
                }
            },
            'data_analysis': {
                'benefit': 'Quantum-accelerated data processing',
                'config': {
                    'computationType': 'OPTIMIZATION',
                    'numQubits': 8,
                    'circuitDepth': 4
                }
            },
            'sentiment_analysis': {
                'benefit': 'Quantum-enhanced sentiment classification',
                'config': {
                    'computationType': 'LOGICAL_INFERENCE',
                    'numQubits': 4,
                    'circuitDepth': 2
                }
            }
        }
        
        benefits = []
        configs = []
        
        for cap in capabilities:
            if cap in quantum_relevant_capabilities:
                benefits.append(quantum_relevant_capabilities[cap]['benefit'])
                configs.append(quantum_relevant_capabilities[cap]['config'])
                
        if benefits:
            return {
                'recommended': True,
                'benefits': benefits,
                'config': {
                    'computationType': 'HYBRID',
                    'numQubits': max(c['numQubits'] for c in configs),
                    'circuitDepth': max(c['circuitDepth'] for c in configs),
                    'optimizationLevel': 2
                }
            }
        
        return {
            'recommended': False,
            'benefits': [],
            'config': None
        }

    def _format_understanding_response(self, domain: Dict[str, Any], objectives: List[str],
                                    capabilities: List[str], skills: List[str],
                                    quantum_benefits: Dict[str, Any]) -> str:
        """Format the understanding response message"""
        response_parts = [
            "I've analyzed your request using quantum symbolic reasoning. Here's my understanding:",
            f"\nDomain: {domain['primary'].title()}",
            f"Secondary domains: {', '.join(domain['secondary']).title() if domain['secondary'] else 'None'}",
            f"\nObjectives:",
            *[f"- {obj.replace('_', ' ').title()}" for obj in objectives],
            f"\nRequired Capabilities:",
            *[f"- {cap.replace('_', ' ').title()}" for cap in capabilities],
            f"\nRequired Skills:",
            *[f"- {skill.replace('_', ' ').title()}" for skill in skills]
        ]
        
        if quantum_benefits['recommended']:
            response_parts.extend([
                "\nQuantum Symbolic Benefits:",
                *[f"- {benefit}" for benefit in quantum_benefits['benefits']]
            ])
            
        response_parts.append("\nIs this understanding correct?")
        
        return "\n".join(response_parts)

    def _format_missing_skills_response(self, missing_skills: set, workflow_plan: Dict[str, Any]) -> str:
        """Format the response for missing skills"""
        response_parts = [
            "I've analyzed the required skills and workflow plan.",
            "\nThe following skills need to be created:",
            *[f"- {skill.replace('_', ' ').title()}" for skill in missing_skills],
            "\nWorkflow Plan:",
            *[f"{i+1}. {step['description']}" for i, step in enumerate(workflow_plan['steps'])],
            "\nWould you like to proceed with creating these skills?"
        ]
        
        return "\n".join(response_parts)

    async def _handle_quantum_configuration(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quantum symbolic configuration"""
        try:
            # Initialize quantum computation
            quantum_config = agent_config['quantum_config']
            
            # Broadcast quantum update
            await self.websocket_manager.broadcast(json.dumps({
                'type': 'quantum_update',
                'data': {
                    'progress': 0,
                    'details': 'Initializing quantum symbolic computation...'
                }
            }))
            
            # Create quantum symbolic expressions
            expressions = []
            for capability in agent_config['capabilities']:
                expr = self.quantum_handler._create_capability_expression(
                    capability,
                    quantum_config
                )
                if expr:
                    expressions.append(expr)
                    
                    # Broadcast progress
                    await self.websocket_manager.broadcast(json.dumps({
                        'type': 'quantum_update',
                        'data': {
                            'progress': len(expressions) * 20,
                            'details': f'Created quantum expression for {capability}'
                        }
                    }))
            
            # Update agent configuration
            agent_config['quantum_expressions'] = expressions
            
            return {
                'success': True,
                'response': "Quantum symbolic configuration complete. Ready to create your agent.",
                'agentConfig': agent_config,
                'nextStep': 'configuring_agent'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error in quantum configuration: {str(e)}"
            }

    async def _handle_standard_configuration(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle standard (non-quantum) configuration"""
        return {
            'success': True,
            'response': "All required skills are available. Ready to create your agent.",
            'agentConfig': agent_config,
            'nextStep': 'configuring_agent'
        }

    async def handle_chat_message(self, message: str, current_step: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a chat message in the agent creation process"""
        try:
            if current_step == 'understanding_needs':
                return await self._handle_needs_understanding(message, agent_config)
            elif current_step == 'confirming_steps':
                return await self._handle_steps_confirmation(message, agent_config)
            elif current_step == 'creating_skills':
                return await self._handle_skill_creation(message, agent_config)
            elif current_step == 'configuring_agent':
                return await self._handle_agent_configuration(message, agent_config)
            else:
                return {
                    'success': False,
                    'error': f'Unknown step: {current_step}'
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _handle_needs_understanding(self, message: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process user's initial description of what they want the agent to do"""
        try:
            # Use reasoning engine to analyze request
            analysis = self.reasoning_engine.analyze_request(message)
            
            # Extract domain and objectives
            domain = analysis['understanding']['domain']
            objectives = analysis['understanding']['objectives']
            
            # Determine required capabilities
            capabilities = self._determine_capabilities(domain, objectives)
            
            # Check if quantum symbolic computation would be beneficial
            quantum_benefits = self._analyze_quantum_benefits(capabilities)
            
            # Update agent config
            agent_config.update({
                'description': message,
                'domain': domain,
                'objectives': objectives,
                'capabilities': capabilities,
                'quantum_symbolic': quantum_benefits['recommended'],
                'quantum_config': quantum_benefits['config'] if quantum_benefits['recommended'] else None
            })
            
            # Analyze required skills
            required_skills = await self._analyze_required_skills(message, capabilities)
            agent_config['required_skills'] = required_skills
            
            # Prepare detailed understanding response
            response = self._format_understanding_response(
                domain=domain,
                objectives=objectives,
                capabilities=capabilities,
                skills=required_skills,
                quantum_benefits=quantum_benefits
            )
            
            # Broadcast reasoning progress
            await self.websocket_manager.broadcast(json.dumps({
                'type': 'reasoning_progress',
                'data': {
                    'thought': 'Analyzing request using quantum symbolic reasoning...',
                    'details': analysis['thought_process']
                }
            }))
            
            return {
                'success': True,
                'response': response,
                'agentConfig': agent_config,
                'nextStep': 'confirming_steps',
                'options': ['Yes, that\'s correct', 'No, let me clarify']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error analyzing request: {str(e)}"
            }

    async def _handle_steps_confirmation(self, message: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user's confirmation or refinement of the proposed steps"""
        try:
            if self._is_confirmation(message):
                # Use reasoning engine to plan steps
                workflow_plan = self.reasoning_engine.plan_workflow(
                    domain=agent_config['domain'],
                    objectives=agent_config['objectives'],
                    capabilities=agent_config['capabilities']
                )
                
                # Check existing skills
                existing_skills = set(self.skill_registry.list_skills())
                required_skills = set(agent_config['required_skills'])
                missing_skills = required_skills - existing_skills
                
                # Broadcast skill check
                await self.websocket_manager.broadcast(json.dumps({
                    'type': 'skill_check',
                    'data': {
                        'required': list(required_skills),
                        'existing': list(existing_skills),
                        'missing': list(missing_skills)
                    }
                }))
                
                if missing_skills:
                    # Some skills need to be created
                    agent_config['workflow_plan'] = workflow_plan
                    agent_config['pending_skills'] = list(missing_skills)
                    
                    response = self._format_missing_skills_response(
                        missing_skills=missing_skills,
                        workflow_plan=workflow_plan
                    )
                    
                    return {
                        'success': True,
                        'response': response,
                        'agentConfig': agent_config,
                        'nextStep': 'creating_skills',
                        'options': ['Proceed with skill creation', 'Modify requirements']
                    }
                else:
                    # All skills exist, proceed with quantum configuration if needed
                    if agent_config['quantum_symbolic']:
                        return await self._handle_quantum_configuration(agent_config)
                    else:
                        return await self._handle_standard_configuration(agent_config)
            else:
                # User wants to refine the understanding
                return await self._handle_needs_understanding(message, agent_config)
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Error confirming steps: {str(e)}"
            }

    async def _handle_skill_creation(self, message: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the creation of new skills"""
        if not agent_config.get('pending_skills'):
            # No more skills to create
            return {
                'success': True,
                'response': "All skills have been created. Let's configure your agent.",
                'agentConfig': agent_config,
                'nextStep': 'configuring_agent'
            }

        current_skill = agent_config['pending_skills'][0]
        
        # Create skill configuration
        skill_config = await self._create_skill_config(current_skill, message)
        
        # Validate skill configuration
        validation_result = validate_skill_config(skill_config)
        if not validation_result['valid']:
            return {
                'success': False,
                'response': f"Error creating skill: {validation_result['error']}",
                'agentConfig': agent_config
            }

        # Store skill for admin approval
        skill_id = f"{current_skill.lower().replace(' ', '_')}_{len(self.pending_skills)}"
        self.pending_skills[skill_id] = skill_config

        # Update agent config
        agent_config['pending_skills'] = agent_config['pending_skills'][1:]
        
        if agent_config['pending_skills']:
            # More skills to create
            next_skill = agent_config['pending_skills'][0]
            response = (
                f"Skill '{current_skill}' has been created and is pending admin approval.\n\n"
                f"Let's create the next skill: {next_skill}"
            )
            return {
                'success': True,
                'response': response,
                'agentConfig': agent_config,
                'nextStep': 'creating_skills'
            }
        else:
            # All skills created
            response = (
                f"Skill '{current_skill}' has been created and is pending admin approval.\n\n"
                f"All required skills have been created. We'll proceed with configuring your agent "
                f"once the skills are approved."
            )
            return {
                'success': True,
                'response': response,
                'agentConfig': agent_config,
                'nextStep': 'configuring_agent'
            }

    async def _handle_agent_configuration(self, message: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the final agent configuration and creation"""
        try:
            # Create the agent
            agent = await self.agent_builder.create_agent(
                name=agent_config.get('name', 'New Agent'),
                description=agent_config['description'],
                skills=agent_config['required_skills'],
                config=agent_config.get('config', {})
            )

            response = (
                f"Your agent has been created successfully!\n\n"
                f"Name: {agent.name}\n"
                f"Type: {agent.type}\n"
                f"Skills: {', '.join(agent.skills)}\n\n"
                f"You can now use your agent in the workspace."
            )

            return {
                'success': True,
                'response': response,
                'agentConfig': agent.to_dict(),
                'nextStep': 'agent_created'
            }

        except Exception as e:
            return {
                'success': False,
                'response': f"Error creating agent: {str(e)}",
                'agentConfig': agent_config
            }

    async def _analyze_required_skills(self, message: str, capabilities: List[str]) -> List[str]:
        """Analyze message and capabilities to determine required skills"""
        skills = []
        
        # Map capabilities to skills
        capability_skill_map = {
            'content_analysis': ['content_creation', 'document_management'],
            'engagement_monitoring': ['social_media_management'],
            'query_understanding': ['rag_skill'],
            'response_generation': ['content_creation'],
            'data_analysis': ['analytics_skill'],
            'pattern_recognition': ['analytics_skill'],
            'conversation_management': ['chat_skill'],
            'sentiment_analysis': ['social_media_management']
        }
        
        # Add skills based on capabilities
        for capability in capabilities:
            if capability in capability_skill_map:
                skills.extend(capability_skill_map[capability])
        
        # Add skills based on keywords in message
        keywords = {
            'email': 'email_management',
            'schedule': 'meeting_management',
            'file': 'file_operations',
            'browse': 'browser_management',
            'install': 'software_management',
            'system': 'system_maintenance',
            'document': 'document_management',
            'shop': 'online_shopping',
            'social': 'social_media_management'
        }

        message_lower = message.lower()
        for keyword, skill in keywords.items():
            if keyword in message_lower:
                skills.append(skill)

        return list(set(skills)) or ['miscellaneous_tasks']  # Remove duplicates, default to miscellaneous if no skills matched

    async def _create_skill_config(self, skill_name: str, description: str) -> Dict[str, Any]:
        """Create a configuration for a new skill"""
        return {
            'name': skill_name,
            'description': description,
            'version': '1.0.0',
            'author': 'AI Agent Builder',
            'capabilities': [],  # Will be filled in by admin during approval
            'requirements': [],  # Will be filled in by admin during approval
            'config_schema': {},  # Will be filled in by admin during approval
            'pending_approval': True
        }

    def _is_confirmation(self, message: str) -> bool:
        """Check if a message is a confirmation"""
        positive_words = {'yes', 'yeah', 'yep', 'correct', 'right', 'ok', 'okay', 'sure', 'good', 'perfect'}
        return any(word in message.lower().split() for word in positive_words)

    def get_pending_skills(self) -> Dict[str, Dict[str, Any]]:
        """Get all skills pending admin approval"""
        return self.pending_skills

    def approve_skill(self, skill_id: str, admin_config: Dict[str, Any]) -> bool:
        """Approve a pending skill with admin-provided configuration"""
        if skill_id not in self.pending_skills:
            return False

        skill_config = self.pending_skills[skill_id]
        skill_config.update(admin_config)
        skill_config['pending_approval'] = False

        # Register the approved skill
        self.skill_registry.register_skill(skill_config)

        # Remove from pending
        del self.pending_skills[skill_id]

        return True

    def reject_skill(self, skill_id: str, reason: str) -> bool:
        """Reject a pending skill"""
        if skill_id not in self.pending_skills:
            return False

        del self.pending_skills[skill_id]
        return True
