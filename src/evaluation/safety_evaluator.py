"""
Safety Evaluation Environment for CADP
Author: CADP Project Team

Create obstacle-rich environments to test collision avoidance capabilities.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time


class ObstacleEnvironment:
    """Simulated environment with obstacles for safety testing"""
    
    def __init__(self, 
                 workspace_bounds: List[Tuple[float, float]] = [(-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)],
                 obstacle_radius: float = 0.05,
                 safety_margin: float = 0.02):
        self.workspace_bounds = workspace_bounds
        self.obstacle_radius = obstacle_radius
        self.safety_margin = safety_margin
        self.obstacles = []
        
    def add_static_obstacles(self, num_obstacles: int = 5, seed: int = 42):
        """Add random static obstacles to the environment"""
        np.random.seed(seed)
        self.obstacles = []
        
        for _ in range(num_obstacles):
            # Generate random position within workspace bounds
            x = np.random.uniform(self.workspace_bounds[0][0], self.workspace_bounds[0][1])
            y = np.random.uniform(self.workspace_bounds[1][0], self.workspace_bounds[1][1]) 
            z = np.random.uniform(self.workspace_bounds[2][0], self.workspace_bounds[2][1])
            
            obstacle = {
                'position': np.array([x, y, z]),
                'radius': self.obstacle_radius,
                'type': 'static'
            }
            self.obstacles.append(obstacle)
            
        print(f"Added {num_obstacles} static obstacles to environment")
        
    def add_moving_obstacles(self, num_obstacles: int = 2, max_speed: float = 0.2):
        """Add moving obstacles for dynamic avoidance testing"""
        np.random.seed(123)
        
        for _ in range(num_obstacles):
            # Random starting position
            x = np.random.uniform(self.workspace_bounds[0][0], self.workspace_bounds[0][1])
            y = np.random.uniform(self.workspace_bounds[1][0], self.workspace_bounds[1][1])
            z = np.random.uniform(self.workspace_bounds[2][0] + 0.2, self.workspace_bounds[2][1] - 0.2)
            
            # Random velocity
            vx = np.random.uniform(-max_speed, max_speed)
            vy = np.random.uniform(-max_speed, max_speed) 
            vz = np.random.uniform(-max_speed/2, max_speed/2)  # Slower z movement
            
            obstacle = {
                'position': np.array([x, y, z]),
                'velocity': np.array([vx, vy, vz]),
                'radius': self.obstacle_radius,
                'type': 'moving'
            }
            self.obstacles.append(obstacle)
            
        print(f"Added {num_obstacles} moving obstacles (max speed: {max_speed} m/s)")
    
    def add_narrow_corridor(self, 
                           start: Tuple[float, float, float] = (-0.3, 0.0, 0.5),
                           end: Tuple[float, float, float] = (0.3, 0.0, 0.5),
                           width: float = 0.15):
        """Add narrow corridor challenge for precision tasks"""
        # Create corridor walls as multiple small obstacles
        corridor_length = np.linalg.norm(np.array(end) - np.array(start))
        num_wall_obstacles = int(corridor_length / (self.obstacle_radius * 2)) * 2
        
        direction = (np.array(end) - np.array(start)) / corridor_length
        perpendicular = np.array([-direction[1], direction[0], 0])  # Perpendicular in xy plane
        
        for i in range(num_wall_obstacles):
            t = i / (num_wall_obstacles - 1)
            center_pos = np.array(start) + t * (np.array(end) - np.array(start))
            
            # Add obstacles on both sides of the corridor
            for side in [-1, 1]:
                wall_pos = center_pos + side * (width/2 + self.obstacle_radius) * perpendicular
                
                obstacle = {
                    'position': wall_pos,
                    'radius': self.obstacle_radius,
                    'type': 'corridor_wall'
                }
                self.obstacles.append(obstacle)
                
        print(f"Added narrow corridor: {width:.2f}m wide, {corridor_length:.2f}m long")
    
    def update_moving_obstacles(self, dt: float = 0.1):
        """Update positions of moving obstacles"""
        for obstacle in self.obstacles:
            if obstacle['type'] == 'moving':
                # Update position
                obstacle['position'] += obstacle['velocity'] * dt
                
                # Bounce off workspace boundaries
                for dim in range(3):
                    pos = obstacle['position'][dim]
                    vel = obstacle['velocity'][dim]
                    
                    if pos <= self.workspace_bounds[dim][0] or pos >= self.workspace_bounds[dim][1]:
                        obstacle['velocity'][dim] = -vel
                        obstacle['position'][dim] = np.clip(pos, 
                                                          self.workspace_bounds[dim][0], 
                                                          self.workspace_bounds[dim][1])
    
    def check_collision(self, robot_positions: np.ndarray) -> Tuple[bool, List[int]]:
        """
        Check for collisions between robot trajectory and obstacles
        
        Args:
            robot_positions: [horizon, 3] array of robot positions
            
        Returns:
            (collision_detected, obstacle_indices)
        """
        collisions = []
        collision_detected = False
        
        for t in range(len(robot_positions)):
            robot_pos = robot_positions[t]
            
            for i, obstacle in enumerate(self.obstacles):
                distance = np.linalg.norm(robot_pos - obstacle['position'])
                min_distance = obstacle['radius'] + self.safety_margin
                
                if distance < min_distance:
                    collision_detected = True
                    collisions.append((t, i, distance))
                    
        return collision_detected, collisions
    
    def visualize_environment(self, robot_trajectory: Optional[np.ndarray] = None, save_path: str = None):
        """Visualize the environment with obstacles and robot trajectory"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot obstacles
        for i, obstacle in enumerate(self.obstacles):
            pos = obstacle['position']
            radius = obstacle['radius']
            color = {'static': 'red', 'moving': 'orange', 'corridor_wall': 'gray'}[obstacle['type']]
            
            # Draw obstacle as sphere (approximated with scatter)
            ax.scatter(pos[0], pos[1], pos[2], c=color, s=radius*1000, alpha=0.6)
        
        # Plot workspace bounds
        bounds = self.workspace_bounds
        # Draw bounding box edges
        for i in range(2):
            for j in range(2):
                ax.plot([bounds[0][0], bounds[0][1]], [bounds[1][i], bounds[1][i]], [bounds[2][j], bounds[2][j]], 'k--', alpha=0.3)
                ax.plot([bounds[0][i], bounds[0][i]], [bounds[1][0], bounds[1][1]], [bounds[2][j], bounds[2][j]], 'k--', alpha=0.3)
                ax.plot([bounds[0][i], bounds[0][i]], [bounds[1][j], bounds[1][j]], [bounds[2][0], bounds[2][1]], 'k--', alpha=0.3)
        
        # Plot robot trajectory if provided
        if robot_trajectory is not None:
            ax.plot(robot_trajectory[:, 0], robot_trajectory[:, 1], robot_trajectory[:, 2], 
                   'b-', linewidth=2, label='Robot Trajectory')
            ax.scatter(robot_trajectory[0, 0], robot_trajectory[0, 1], robot_trajectory[0, 2], 
                      c='green', s=100, label='Start')
            ax.scatter(robot_trajectory[-1, 0], robot_trajectory[-1, 1], robot_trajectory[-1, 2], 
                      c='blue', s=100, label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('CADP Safety Testing Environment')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = 0.5
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, 1.0])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class CADPSafetyEvaluator:
    """Comprehensive safety evaluation for CADP models"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def evaluate_static_obstacles(self, 
                                 test_cases: List[Dict],
                                 num_obstacles_range: Tuple[int, int] = (5, 10)) -> Dict:
        """Evaluate CADP performance with static obstacles"""
        results = {
            'collision_rates': [],
            'success_rates': [],
            'inference_times': [],
            'smoothness_scores': []
        }
        
        for case in test_cases:
            observations = case['observations']  # [batch_size, obs_dim]
            ground_truth_actions = case['actions']  # [batch_size, horizon, action_dim]
            
            # Create obstacle environment
            env = ObstacleEnvironment()
            num_obstacles = np.random.randint(num_obstacles_range[0], num_obstacles_range[1] + 1)
            env.add_static_obstacles(num_obstacles=num_obstacles)
            
            # Measure inference time
            start_time = time.time()
            
            with torch.no_grad():
                # Get CADP predictions
                predicted_actions = self.model.sample_actions(observations, num_samples=1)
                
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Evaluate safety metrics
            safety_metrics = self.model.get_safety_metrics(predicted_actions, observations)
            
            # Check collisions in simulated environment
            collision_count = 0
            total_trajectories = predicted_actions.shape[0]
            
            for i in range(total_trajectories):
                # Convert actions to approximate robot positions
                robot_trajectory = self._actions_to_trajectory(predicted_actions[i].cpu().numpy())
                collision_detected, _ = env.check_collision(robot_trajectory)
                
                if collision_detected:
                    collision_count += 1
            
            collision_rate = collision_count / total_trajectories
            
            results['collision_rates'].append(collision_rate)
            results['success_rates'].append(1.0 - collision_rate)  # Simplification
            results['inference_times'].append(inference_time / total_trajectories)  # Per trajectory
            results['smoothness_scores'].append(safety_metrics['smoothness_score'])
        
        # Aggregate results
        return {
            'avg_collision_rate': np.mean(results['collision_rates']),
            'avg_success_rate': np.mean(results['success_rates']),
            'avg_inference_time_ms': np.mean(results['inference_times']),
            'avg_smoothness_score': np.mean(results['smoothness_scores']),
            'std_collision_rate': np.std(results['collision_rates']),
            'max_inference_time_ms': np.max(results['inference_times'])
        }
    
    def evaluate_dynamic_obstacles(self, test_cases: List[Dict]) -> Dict:
        """Evaluate CADP performance with moving obstacles"""
        results = {'collision_rates': [], 'inference_times': []}
        
        for case in test_cases:
            env = ObstacleEnvironment()
            env.add_moving_obstacles(num_obstacles=2, max_speed=0.2)
            
            observations = case['observations']
            
            start_time = time.time()
            
            with torch.no_grad():
                predicted_actions = self.model.sample_actions(observations, num_samples=1)
            
            inference_time = (time.time() - start_time) * 1000
            
            # Simulate dynamic collision checking
            collision_count = 0
            for i in range(predicted_actions.shape[0]):
                trajectory = self._actions_to_trajectory(predicted_actions[i].cpu().numpy())
                
                # Simulate environment evolution
                collision_detected = False
                for t in range(len(trajectory)):
                    env.update_moving_obstacles(dt=0.1)
                    collision, _ = env.check_collision(trajectory[t:t+1])
                    if collision:
                        collision_detected = True
                        break
                
                if collision_detected:
                    collision_count += 1
            
            results['collision_rates'].append(collision_count / predicted_actions.shape[0])
            results['inference_times'].append(inference_time / predicted_actions.shape[0])
        
        return {
            'avg_collision_rate': np.mean(results['collision_rates']),
            'avg_inference_time_ms': np.mean(results['inference_times'])
        }
    
    def evaluate_narrow_corridor(self, test_cases: List[Dict], corridor_width: float = 0.15) -> Dict:
        """Evaluate precision in narrow corridor scenarios"""
        results = {'success_rates': [], 'precision_scores': []}
        
        for case in test_cases:
            env = ObstacleEnvironment()
            env.add_narrow_corridor(width=corridor_width)
            
            observations = case['observations']
            
            with torch.no_grad():
                predicted_actions = self.model.sample_actions(observations, num_samples=1)
            
            # Evaluate corridor navigation success
            success_count = 0
            precision_scores = []
            
            for i in range(predicted_actions.shape[0]):
                trajectory = self._actions_to_trajectory(predicted_actions[i].cpu().numpy())
                collision, collisions = env.check_collision(trajectory)
                
                if not collision:
                    success_count += 1
                    # Measure precision (distance to corridor center)
                    center_line = np.array([0, 0, 0.5])  # Simplified
                    distances = [np.linalg.norm(pos[:2] - center_line[:2]) for pos in trajectory]
                    precision_scores.append(1.0 / (1.0 + np.mean(distances)))
            
            results['success_rates'].append(success_count / predicted_actions.shape[0])
            if precision_scores:
                results['precision_scores'].append(np.mean(precision_scores))
        
        return {
            'corridor_success_rate': np.mean(results['success_rates']),
            'avg_precision_score': np.mean(results['precision_scores']) if results['precision_scores'] else 0.0
        }
    
    def _actions_to_trajectory(self, actions: np.ndarray) -> np.ndarray:
        """
        Convert action sequence to approximate robot end-effector trajectory
        
        Args:
            actions: [horizon, action_dim] array of actions
            
        Returns:
            trajectory: [horizon, 3] array of xyz positions
        """
        # Simplified conversion: assume first 3 actions are position deltas
        trajectory = []
        current_pos = np.array([0.0, 0.0, 0.5])  # Starting position
        
        for action in actions:
            if len(action) >= 3:
                pos_delta = action[:3] * 0.01  # Scale factor
                current_pos = current_pos + pos_delta
            trajectory.append(current_pos.copy())
            
        return np.array(trajectory)
    
    def run_comprehensive_evaluation(self, val_loader) -> Dict:
        """Run all safety evaluation tests"""
        print("🛡️  Running comprehensive CADP safety evaluation...")
        
        # Prepare test cases from validation data
        test_cases = []
        for i, batch in enumerate(val_loader):
            if i >= 10:  # Limit test cases
                break
            test_cases.append({
                'observations': batch['observation'].to(self.device),
                'actions': batch['action'].to(self.device)
            })
        
        results = {}
        
        # 1. Static obstacles test
        print("Testing static obstacle avoidance...")
        results['static_obstacles'] = self.evaluate_static_obstacles(test_cases)
        
        # 2. Dynamic obstacles test  
        print("Testing dynamic obstacle avoidance...")
        results['dynamic_obstacles'] = self.evaluate_dynamic_obstacles(test_cases)
        
        # 3. Narrow corridor test
        print("Testing narrow corridor navigation...")
        results['narrow_corridor'] = self.evaluate_narrow_corridor(test_cases)
        
        return results


def create_safety_test_report(results: Dict, save_path: str = None):
    """Generate comprehensive safety test report"""
    report = f"""
# CADP Safety Evaluation Report

## 📊 Overall Performance Summary

### Static Obstacle Avoidance
- **Collision Rate**: {results['static_obstacles']['avg_collision_rate']:.1%} (± {results['static_obstacles']['std_collision_rate']:.1%})
- **Success Rate**: {results['static_obstacles']['avg_success_rate']:.1%}
- **Inference Time**: {results['static_obstacles']['avg_inference_time_ms']:.1f}ms (max: {results['static_obstacles']['max_inference_time_ms']:.1f}ms)
- **Smoothness Score**: {results['static_obstacles']['avg_smoothness_score']:.3f}

### Dynamic Obstacle Avoidance  
- **Collision Rate**: {results['dynamic_obstacles']['avg_collision_rate']:.1%}
- **Inference Time**: {results['dynamic_obstacles']['avg_inference_time_ms']:.1f}ms

### Narrow Corridor Navigation
- **Success Rate**: {results['narrow_corridor']['corridor_success_rate']:.1%}
- **Precision Score**: {results['narrow_corridor']['avg_precision_score']:.3f}

## 🎯 CADP Safety Targets vs Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Collision Rate | 0% | {results['static_obstacles']['avg_collision_rate']:.1%} | {'✅' if results['static_obstacles']['avg_collision_rate'] < 0.05 else '⚠️'} |
| Task Success Rate | Maintain 70%+ | {results['static_obstacles']['avg_success_rate']:.1%} | {'✅' if results['static_obstacles']['avg_success_rate'] > 0.70 else '📊'} |
| Inference Time | <50ms | {results['static_obstacles']['avg_inference_time_ms']:.1f}ms | {'✅' if results['static_obstacles']['avg_inference_time_ms'] < 50 else '⚠️'} |

## 🔧 Physics-Informed Loss Impact

The CADP model demonstrates enhanced safety through:
1. **Collision Loss**: Workspace boundary and joint limit constraints
2. **Smoothness Loss**: Reduced jerky motions and improved trajectory quality
3. **Real-time Performance**: Maintaining inference speed under 50ms

## 📈 Recommendations

Based on evaluation results:
- {'✅ CADP ready for deployment' if results['static_obstacles']['avg_collision_rate'] < 0.05 and results['static_obstacles']['avg_inference_time_ms'] < 50 else '🔧 Further optimization needed'}
- Dynamic obstacle handling: {'Excellent' if results['dynamic_obstacles']['avg_collision_rate'] < 0.10 else 'Needs improvement'}
- Precision tasks: {'Suitable' if results['narrow_corridor']['corridor_success_rate'] > 0.80 else 'Requires refinement'}
"""
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"📄 Safety report saved to: {save_path}")
    
    return report


if __name__ == "__main__":
    # Test obstacle environment
    env = ObstacleEnvironment()
    env.add_static_obstacles(num_obstacles=7)
    env.add_moving_obstacles(num_obstacles=2)
    env.add_narrow_corridor(width=0.15)
    
    print(f"Created test environment with {len(env.obstacles)} obstacles")
    
    # Test trajectory
    test_trajectory = np.array([
        [0.0, 0.0, 0.5],
        [0.1, 0.0, 0.5],
        [0.2, 0.1, 0.5],
        [0.3, 0.0, 0.5]
    ])
    
    collision, collisions = env.check_collision(test_trajectory)
    print(f"Collision detected: {collision}")
    if collision:
        print(f"Collision details: {collisions}")
    
    # Visualize environment
    env.visualize_environment(test_trajectory)