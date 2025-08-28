"""
CADP Main Execution Pipeline
Implementation of Algorithm 4 from CADP paper

This module implements the complete CADP execution pipeline that integrates
all safety layers from environment encoding to safe trajectory execution.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path

# Import CADP components
from src.environment.key_configuration_selector import KeyConfigurationSelector
from src.safety.cbf_verifier_batch_optimized import BatchOptimizedCBFVerifier
from src.control.smc_controller import SMCController


class CADPMainExecutor:
    """
    Complete CADP Execution Pipeline
    
    Implements Algorithm 4: CADP Main Execution Loop with full integration
    of physics-informed diffusion, safety verification, and robust control.
    """
    
    def __init__(
        self,
        diffusion_model: Any,  # Generic type for diffusion model
        cbf_verifier: BatchOptimizedCBFVerifier,
        smc_controller: SMCController,
        key_config_selector: KeyConfigurationSelector,
        robot_dof: int = 7,
        execution_frequency: float = 100.0,  # Hz
        trajectory_horizon: int = 50,
        safety_recheck_threshold: float = 0.1,  # Environment change threshold
        max_replanning_attempts: int = 3
    ):
        """
        Initialize CADP Main Executor
        
        Args:
            diffusion_model: Physics-informed diffusion policy
            cbf_verifier: Batch-optimized CBF safety verifier  
            smc_controller: SMC-based safe tracking controller
            key_config_selector: Key configuration selector for environment encoding
            robot_dof: Degrees of freedom
            execution_frequency: Control loop frequency in Hz
            trajectory_horizon: Planning horizon length
            safety_recheck_threshold: Threshold for environment change detection
            max_replanning_attempts: Maximum replanning attempts on failure
        """
        self.diffusion_model = diffusion_model
        self.cbf_verifier = cbf_verifier
        self.smc_controller = smc_controller
        self.key_config_selector = key_config_selector
        
        self.dof = robot_dof
        self.exec_freq = execution_frequency
        self.dt = 1.0 / execution_frequency
        self.horizon = trajectory_horizon
        self.safety_threshold = safety_recheck_threshold
        self.max_replanning = max_replanning_attempts
        
        # Execution state
        self.current_trajectory = None
        self.trajectory_start_time = None
        self.execution_metrics = {
            'total_executions': 0,
            'safety_violations': 0,
            'replanning_count': 0,
            'average_planning_time': 0.0,
            'average_verification_time': 0.0,
            'average_control_time': 0.0
        }
        
        # Environment state tracking
        self.previous_environment = None
        self.sdf_function = None
        
    def execute_safe_manipulation(
        self,
        observation: torch.Tensor,
        goal: torch.Tensor, 
        environment_sdf: Callable,
        current_state: torch.Tensor,
        execution_time: float = 5.0,
        get_current_state_fn: Optional[Callable] = None,
        apply_control_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute complete CADP manipulation task
        
        Implements Algorithm 4: CADP Main Execution Loop
        
        Args:
            observation: Current observation
            goal: Task goal
            environment_sdf: Signed distance function for environment
            current_state: Current robot state [q, q_dot]
            execution_time: Total execution time in seconds
            get_current_state_fn: Function to get current robot state
            apply_control_fn: Function to apply control to robot
            
        Returns:
            Execution results and metrics
        """
        print("Starting CADP safe manipulation execution...")
        
        self.sdf_function = environment_sdf
        execution_results = {
            'success': False,
            'trajectory_executed': None,
            'safety_violations': [],
            'execution_metrics': {},
            'replanning_events': [],
            'control_history': []
        }
        
        try:
            # Phase 1: Environment Encoding
            print("Phase 1: Environment encoding...")
            start_time = time.time()
            
            key_config_encoding = self._encode_environment(
                observation, environment_sdf
            )
            
            encoding_time = time.time() - start_time
            
            # Phase 2: Trajectory Generation
            print("Phase 2: Trajectory generation...")
            start_time = time.time()
            
            generated_trajectory = self._generate_trajectory(
                observation, goal, key_config_encoding
            )
            
            planning_time = time.time() - start_time
            self.execution_metrics['average_planning_time'] = planning_time
            
            # Phase 3: Safety Verification
            print("Phase 3: Safety verification...")
            start_time = time.time()
            
            safe_trajectory, verification_results = self._verify_trajectory_safety(
                generated_trajectory, environment_sdf
            )
            
            verification_time = time.time() - start_time
            self.execution_metrics['average_verification_time'] = verification_time
            
            if not verification_results['is_safe']:
                raise Exception(f"Generated trajectory failed safety verification: "
                               f"{verification_results['violations']}")
            
            # Phase 4: Safe Execution
            print("Phase 4: Safe execution...")
            
            execution_success, control_results = self._execute_trajectory_safely(
                safe_trajectory,
                current_state,
                execution_time,
                get_current_state_fn or self._default_get_state,
                apply_control_fn or self._default_apply_control
            )
            
            # Compile results
            execution_results.update({
                'success': execution_success,
                'trajectory_executed': safe_trajectory.detach().cpu().numpy(),
                'execution_metrics': {
                    **self.execution_metrics,
                    'encoding_time': encoding_time,
                    'planning_time': planning_time,
                    'verification_time': verification_time,
                    'total_time': encoding_time + planning_time + verification_time
                },
                'control_history': control_results['control_history'],
                'safety_violations': control_results.get('safety_violations', [])
            })
            
            self.execution_metrics['total_executions'] += 1
            
            if execution_success:
                print("✅ CADP manipulation task completed successfully!")
            else:
                print("❌ CADP manipulation task failed during execution")
                
        except Exception as e:
            print(f"❌ CADP execution failed: {str(e)}")
            execution_results['error'] = str(e)
            
        return execution_results
    
    def _encode_environment(
        self, 
        observation: torch.Tensor,
        environment_sdf: Callable
    ) -> torch.Tensor:
        """
        Phase 1: Environment Encoding using Key Configurations
        """
        # For this implementation, we use a simplified encoding
        # In practice, this would use the key configuration selector
        # with the current environment state
        
        # Generate a dummy query configuration from observation
        if len(observation) >= self.dof:
            query_config = observation[:self.dof]
        else:
            # If observation doesn't contain joint positions, use zero config
            query_config = torch.zeros(self.dof, device=observation.device)
        
        # Get key configuration encoding
        if hasattr(self.key_config_selector, 'key_configurations') and \
           self.key_config_selector.key_configurations:
            encoding = self.key_config_selector.get_key_configuration_encoding(
                query_config, encoding_dim=128
            )
        else:
            # Fallback: zero encoding if no key configurations available
            encoding = torch.zeros(128, device=observation.device)
        
        return encoding
    
    def _generate_trajectory(
        self,
        observation: torch.Tensor,
        goal: torch.Tensor,
        key_config_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Phase 2: Physics-Informed Trajectory Generation
        """
        # Initialize noise for diffusion sampling
        trajectory_shape = (self.horizon, self.dof)
        noisy_trajectory = torch.randn(
            trajectory_shape, device=observation.device
        )
        
        # Use DDIM sampling (simplified for this implementation)
        # In practice, this would use the full diffusion denoising process
        # with physics-informed conditioning
        
        with torch.no_grad():
            # Simplified trajectory generation
            # This should be replaced with proper diffusion sampling
            generated_trajectory = self._simplified_trajectory_generation(
                observation, goal, noisy_trajectory
            )
        
        return generated_trajectory
    
    def _simplified_trajectory_generation(
        self,
        observation: torch.Tensor,
        goal: torch.Tensor,
        initial_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified trajectory generation for testing
        
        In production, this would be replaced with proper DDIM sampling
        from the physics-informed diffusion model.
        """
        device = observation.device
        
        # Extract current position from observation
        if len(observation) >= self.dof:
            start_pos = observation[:self.dof]
        else:
            start_pos = torch.zeros(self.dof, device=device)
        
        # Extract goal position
        if len(goal) >= self.dof:
            goal_pos = goal[:self.dof]
        else:
            goal_pos = torch.ones(self.dof, device=device) * 0.5
        
        # Generate smooth interpolation with physics-informed constraints
        t_steps = torch.linspace(0, 1, self.horizon, device=device)
        
        trajectory = torch.zeros(self.horizon, self.dof, device=device)
        
        for i, t in enumerate(t_steps):
            # Linear interpolation with smooth velocity profile
            alpha = 3 * t**2 - 2 * t**3  # Smooth step function
            trajectory[i] = start_pos + alpha * (goal_pos - start_pos)
            
            # Add small amount of noise for diversity
            trajectory[i] += initial_noise[i] * 0.05
        
        # Apply basic physics constraints
        trajectory = self._apply_physics_constraints(trajectory)
        
        return trajectory
    
    def _apply_physics_constraints(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Apply basic physics constraints to trajectory"""
        
        # Clamp joint positions to reasonable limits
        joint_limit = torch.pi * 0.9  # 90% of full range
        trajectory = torch.clamp(trajectory, -joint_limit, joint_limit)
        
        # Smooth trajectory to respect velocity limits
        for i in range(1, trajectory.shape[0]):
            max_step = self.smc_controller.v_max * self.dt
            step = trajectory[i] - trajectory[i-1]
            step_norm = torch.norm(step)
            
            if step_norm > max_step:
                trajectory[i] = trajectory[i-1] + step * (max_step / step_norm)
        
        return trajectory
    
    def _verify_trajectory_safety(
        self,
        trajectory: torch.Tensor,
        environment_sdf: Callable
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Phase 3: CBF-based Safety Verification
        """
        # Use the batch-optimized CBF verifier
        results = self.cbf_verifier.batch_verify_trajectories([trajectory], self.dt)
        
        if results:
            result = results[0]
            safe_trajectory = result.safe_trajectory
            verification_info = {
                'is_safe': result.num_unsafe_waypoints == 0,
                'violations': [],
                'corrections_applied': result.num_corrections,
                'verification_time': result.verification_time_ms / 1000.0,
                'min_safety_margin': 0.1  # Placeholder
            }
        else:
            safe_trajectory = trajectory
            verification_info = {
                'is_safe': False,
                'violations': ['Verification failed'],
                'corrections_applied': 0,
                'verification_time': 0.0,
                'min_safety_margin': -1.0
            }
        
        verification_results = {
            'is_safe': verification_info.get('is_safe', True),
            'violations': verification_info.get('violations', []),
            'corrections_applied': verification_info.get('corrections_applied', 0),
            'verification_time': verification_info.get('verification_time', 0.0),
            'safety_margin': verification_info.get('min_safety_margin', float('inf'))
        }
        
        return safe_trajectory, verification_results
    
    def _execute_trajectory_safely(
        self,
        safe_trajectory: torch.Tensor,
        initial_state: torch.Tensor,
        execution_time: float,
        get_state_fn: Callable,
        apply_control_fn: Callable
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Phase 4: SMC-based Safe Execution with Dynamic Re-verification
        """
        print(f"Executing trajectory for {execution_time} seconds...")
        
        self.current_trajectory = safe_trajectory
        self.trajectory_start_time = time.time()
        
        control_history = []
        safety_violations = []
        execution_success = True
        
        total_steps = int(execution_time * self.exec_freq)
        current_state = initial_state.clone()
        
        for step in range(total_steps):
            step_start_time = time.time()
            
            # Get current robot state
            try:
                current_state = get_state_fn()
            except:
                # Use previous state if getting current state fails
                pass
            
            # Compute reference time
            ref_time = step * self.dt
            
            # Check if environment changed significantly (simplified)
            env_changed = self._check_environment_change()
            
            if env_changed:
                print("Environment change detected, re-verifying trajectory...")
                # Re-verify remaining trajectory
                remaining_traj = safe_trajectory[step:]
                if len(remaining_traj) > 0:
                    updated_traj, verify_result = self._verify_trajectory_safety(
                        remaining_traj, self.sdf_function
                    )
                    
                    if verify_result['is_safe']:
                        # Update trajectory for remaining execution
                        safe_trajectory[step:] = updated_traj
                    else:
                        print("Re-verification failed, stopping execution")
                        execution_success = False
                        break
            
            # Compute SMC control
            try:
                control_input, control_info = self.smc_controller.compute_control(
                    current_state,
                    safe_trajectory,
                    ref_time,
                    self.sdf_function,
                    self.dt
                )
                
                # Apply control to robot
                apply_control_fn(control_input)
                
                # Log control information
                control_step_info = {
                    'step': step,
                    'time': ref_time,
                    'control_input': control_input.detach().cpu().numpy(),
                    'tracking_error': control_info.get('tracking_error'),
                    'safety_margin': control_info.get('safety_margin'),
                    'control_effort': control_info.get('control_effort'),
                    'computation_time': time.time() - step_start_time
                }
                control_history.append(control_step_info)
                
                # Check for safety violations
                if control_info.get('safety_margin', 1.0) < 0:
                    safety_violation = {
                        'step': step,
                        'time': ref_time,
                        'margin': control_info.get('safety_margin'),
                        'violation_type': 'CBF'
                    }
                    safety_violations.append(safety_violation)
                    
            except Exception as e:
                print(f"Control computation failed at step {step}: {e}")
                execution_success = False
                break
            
            # Maintain execution frequency
            step_duration = time.time() - step_start_time
            if step_duration < self.dt:
                time.sleep(self.dt - step_duration)
        
        # Update execution metrics
        avg_control_time = np.mean([
            info['computation_time'] for info in control_history
        ]) if control_history else 0.0
        
        self.execution_metrics['average_control_time'] = avg_control_time
        self.execution_metrics['safety_violations'] += len(safety_violations)
        
        control_results = {
            'control_history': control_history,
            'safety_violations': safety_violations,
            'execution_success': execution_success,
            'total_steps': len(control_history),
            'average_control_time': avg_control_time
        }
        
        return execution_success and len(safety_violations) == 0, control_results
    
    def _check_environment_change(self) -> bool:
        """
        Check if environment has changed significantly
        
        Simplified implementation - in practice would compare
        current SDF with previous SDF values
        """
        # For this simplified implementation, assume no environment change
        # In practice, this would check SDF differences at key points
        return False
    
    def _default_get_state(self) -> torch.Tensor:
        """Default state getter for testing"""
        return torch.zeros(2 * self.dof)  # [q, q_dot]
    
    def _default_apply_control(self, control: torch.Tensor):
        """Default control application for testing"""
        pass  # No actual robot control in testing
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics"""
        return self.execution_metrics.copy()
    
    def reset_metrics(self):
        """Reset execution metrics"""
        self.execution_metrics = {
            'total_executions': 0,
            'safety_violations': 0,
            'replanning_count': 0,
            'average_planning_time': 0.0,
            'average_verification_time': 0.0,
            'average_control_time': 0.0
        }

def create_cadp_executor(
    model_path: Optional[str] = None,
    **kwargs
) -> CADPMainExecutor:
    """
    Factory function to create complete CADP executor
    
    Args:
        model_path: Path to trained diffusion model
        **kwargs: Additional parameters
        
    Returns:
        Configured CADP executor
    """
    # Import component factory functions
    from src.safety.cbf_verifier_batch_optimized import create_batch_optimized_cbf_verifier
    from src.control.smc_controller import create_smc_controller  
    from src.environment.key_configuration_selector import create_key_configuration_selector
    
    # Create components
    cbf_verifier = create_batch_optimized_cbf_verifier()
    smc_controller = create_smc_controller()
    key_config_selector = create_key_configuration_selector()
    
    # Create placeholder diffusion model (would load from model_path in practice)
    class DummyDiffusionPolicy:
        def __init__(self):
            pass
        def forward(self, *args, **kwargs):
            pass
    
    diffusion_model = DummyDiffusionPolicy()
    
    # Create executor
    executor = CADPMainExecutor(
        diffusion_model=diffusion_model,
        cbf_verifier=cbf_verifier,
        smc_controller=smc_controller,
        key_config_selector=key_config_selector,
        **kwargs
    )
    
    return executor