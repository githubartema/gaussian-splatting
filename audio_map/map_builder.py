import torch
import torchaudio
import numpy as np
import json
import soundfile as sf
import tqdm
from matplotlib import pyplot as plt
import os

class AudioMapBuilder():
    def __init__(
        self,
        left_channel, 
        right_channel, 
        receiver_positions,
        receiver_rotations, 
        points,
        moving_window: int = 30
    ):
        self.left_channel = left_channel
        self.right_channel = right_channel
        self.receiver_positions = receiver_positions
        self.receiver_rotations = receiver_rotations 
        self.points = points  
        self.moving_window = moving_window 

        self.num_steps = len(self.receiver_positions)
        self.step_size = self.left_channel.size(0) // self.num_steps

    @staticmethod
    def compute_rms(tensor):
        """
        Compute the RMS intensity of a given tensor.
        
        Args:
        tensor (torch.Tensor): The audio samples.
        
        Returns:
        torch.Tensor: The RMS intensity.
        """
        
        return torch.sqrt(torch.mean(tensor ** 2))
    
    def determine_point_side(self, receiver_position, receiver_rotation, point):
        """
        Determine if the point is on the left or right of the receiver based on its orientation.
        
        Parameters:
        receiver_position (tuple): The (x, y) position of the receiver.
        receiver_rotation (float): The rotation angle of the receiver in degrees.
        point (tuple): The (x, y) position of the point to be checked.
        
        Returns:
        str: "l" if the point is on the left, "r" if the point is on the right, "inline" if directly ahead.
        """
        # Convert orientation angle to radians and adjust for Y axis inversion
        receiver_rotation %= 360

        theta = np.radians(receiver_rotation)
        
        # Calculate direction vector from orientation angle (accounting for Y-axis inversion)
        d = np.array([np.cos(theta), np.sin(theta)])
        
        # Compute vector from receiver to point
        r = np.array(receiver_position)
        o = np.array(point)
        
        v = o - r
        
        # Calculate the determinant (simplified 2D cross product)
        det = d[0] * v[1] - d[1] * v[0]
        
        if det < 0:
            return "r"  # In screen coordinates, a negative determinant indicates the point is to the right
        elif det > 0:
            return "l"  # In screen coordinates, a positive determinant indicates the point is to the left
        else:
            return "inline"
        
    def indicator_function(
            self, 
            louder_side: str,
            receiver_position: np.ndarray, 
            receiver_rotation: np.ndarray,
            point: np.ndarray
        ) -> bool:

        # Determine the orientation of the point relative to the receiver
        point_side = self.determine_point_side(
            receiver_position, 
            receiver_rotation, 
            point
        )

        # Return True if the orientation matches the specified side, False otherwise
        return point_side == louder_side
        
    @staticmethod
    def moving_average(data, window_size):
        """
        Apply a moving average filter to the data.
        
        Args:
        data (np.ndarray): The input data to filter.
        window_size (int): The size of the moving window.
        
        Returns:
        np.ndarray: The filtered data with initial values unfiltered.
        """
        filtered_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
        initial_values = data[:window_size - 1]  # Keep the initial values unfiltered
        return np.concatenate((initial_values, filtered_data))
    
    def compute_confidence_coefficients(self):
        """
        Compute the confidence coefficients for given left and right channel audio data.
        """
        
        louder_sides = []
        confidence_coefficients = np.zeros(len(self.points))
        indicator_values = np.zeros((self.num_steps, len(self.points)))
        rms_left_history, rms_right_history = [], []

        for t in range(self.num_steps):
            start_idx = t * self.step_size
            end_idx = (t + 1) * self.step_size

            # Extract the segment for the current time step
            left_segment = self.left_channel[start_idx:end_idx]
            right_segment = self.right_channel[start_idx:end_idx]

            # Compute RMS intensity for both channels
            rms_left = self.compute_rms(left_segment)
            rms_right = self.compute_rms(right_segment)
            rms_left_history.append(rms_left)
            rms_right_history.append(rms_right)

        rms_left_history = self.moving_average(np.array(rms_left_history), self.moving_window)
        rms_right_history = self.moving_average(np.array(rms_right_history), self.moving_window)
        
        for t, (receiver_position, receiver_rotation) in enumerate(zip(self.receiver_positions, self.receiver_rotations)):
            
            louder_side = 'l' if rms_left_history[t] > rms_right_history[t] else 'r'
            louder_sides.append(louder_side)

            # Calculate weight
            WEIGHT = (
                abs(rms_left_history[t] - rms_right_history[t]) 
                / (max(rms_left_history[t], rms_right_history[t]) + 1e-6)
            )
            
            for i, point in enumerate(self.points):
                indicator_values[t, i] = self.indicator_function(louder_side, receiver_position, receiver_rotation, point)
                
                # if np.linalg.norm(point - receiver_position) < 10:
                confidence_coefficients[i] += WEIGHT * indicator_values[t, i]

        confidence_coefficients = np.array(confidence_coefficients) / np.max(confidence_coefficients)

        return confidence_coefficients, louder_sides, rms_left_history, rms_right_history, indicator_values
    
