"""
Asynchronous Sensor Event Simulator based on Python heapq Priority Queue.
Simulates non-synchronized, multi-rate sensor observation streams.
"""

from __future__ import annotations
import heapq
from typing import List, Iterator, Optional, Callable
import numpy as np

from .sensors import BaseSensor, SensorMeasurement
from .dynamics import ConstantVelocityTrajectory


class AsyncSensorEventSimulator:
    """
    Event-driven priority queue simulator for multi-sensor airborne tracking.

    Manages independent sensor clocks and interleaves events strictly in chronological
    order using a min-heap structure (`heapq`).
    """

    def __init__(self, trajectory: ConstantVelocityTrajectory) -> None:
        """
        Args:
            trajectory: Ground truth trajectory generator for true target states.
        """
        self.trajectory = trajectory
        self._sensors: List[tuple[BaseSensor, float]] = []  # (sensor, initial_phase_offset)
        self._event_queue: List[SensorMeasurement] = []

    def register_sensor(self, sensor: BaseSensor, phase_offset: float = 0.0) -> None:
        """
        Register a sensor with an optional initial phase offset to model clock desynchronization.

        Args:
            sensor: Sensor instance (Radar, Camera, etc.).
            phase_offset: Initial time delay/offset in seconds (default 0.0).
        """
        self._sensors.append((sensor, float(phase_offset)))

    def populate_events(
        self,
        t_start: float,
        t_end: float,
        uav_position_fn: Optional[Callable[[float], np.ndarray]] = None,
        jitter_std: float = 0.0,
    ) -> int:
        """
        Populate the priority queue with all sensor measurements over [t_start, t_end].

        Args:
            t_start: Simulation start epoch in seconds.
            t_end: Simulation end epoch in seconds.
            uav_position_fn: Optional function mapping t -> 3D UAV position vector.
                             If None, UAV is assumed stationary at [0, 0, 0].
            jitter_std: Standard deviation of clock timing jitter (seconds). Default 0.0.

        Returns:
            int: Total number of discrete events enqueued.
        """
        self._event_queue.clear()

        for sensor, offset in self._sensors:
            dt = sensor.period
            # Starting time for this sensor taking into account phase offset
            curr_t = t_start + offset
            while curr_t <= t_end:
                # Add optional timestamp jitter
                t_meas = curr_t
                if jitter_std > 0.0:
                    t_meas += float(sensor.rng.normal(0.0, jitter_std))
                    t_meas = max(t_start, t_meas)

                # Ground truth target position at measurement time
                tgt_state = self.trajectory.get_state(t_meas)
                uav_pos = uav_position_fn(t_meas) if uav_position_fn is not None else np.zeros(3)

                measurement = sensor.measure(timestamp=t_meas, target_pos=tgt_state.position, uav_pos=uav_pos)
                heapq.heappush(self._event_queue, measurement)

                curr_t += dt

        return len(self._event_queue)

    def has_next(self) -> bool:
        """Checks if there are remaining sensor events in the queue."""
        return len(self._event_queue) > 0

    def pop_next(self) -> SensorMeasurement:
        """Extracts the chronologically earliest sensor measurement."""
        if not self._event_queue:
            raise IndexError("pop from empty event queue")
        return heapq.heappop(self._event_queue)

    def __iter__(self) -> Iterator[SensorMeasurement]:
        """Provides an iterable stream over the chronologically sorted sensor events."""
        while self.has_next():
            yield self.pop_next()

    def __len__(self) -> int:
        """Returns the number of events currently queued."""
        return len(self._event_queue)
