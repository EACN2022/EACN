from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.common import observation
from highway_env.vehicle.controller import MDPVehicle
from highway_env.road.lane import AbstractLane
from typing import List, Dict
from gym import spaces
import numpy as np
from highway_env import utils
import pandas as pd




class KinematicObservation(observation.ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = False,
                 observe_intentions: bool = False,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count,len(self.features)), low=-1, high=1, dtype=np.float32)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.
        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * MDPVehicle.SPEED_MAX, 5.0 * MDPVehicle.SPEED_MAX],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-MDPVehicle.SPEED_MAX, MDPVehicle.SPEED_MAX],
                "vy": [-MDPVehicle.SPEED_MAX, MDPVehicle.SPEED_MAX]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")

        self.env.v_in_obs = dict()
        self.env.v_in_obs[0] = self.observer_vehicle
        for i, v in enumerate(close_vehicles[-self.vehicles_count + 1:]):
            self.env.v_in_obs[i+1] = v
        i += 1
        while i < self.vehicles_count:
            self.env.v_in_obs[i] = None
            i += 1

        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = df.append(pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs






observation.KinematicObservation = KinematicObservation


def observation_factory(env: 'AbstractEnv', config: dict) -> observation.ObservationType:
    if config["type"] == "TimeToCollision":
        return observation.TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return observation.KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return observation.OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return observation.KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return observation.GrayscaleObservation(env, config)
    elif config["type"] == "AttributesObservation":
        return observation.AttributesObservation(env, **config)
    elif config["type"] == "MultiAgentObservation":
        return observation.MultiAgentObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")

observation.observation_factory = observation_factory





def attack_env(self, adv_obs):

    adv_obs = adv_obs.copy().reshape((-1,5))

    feat_range = self.observation_type.features_range
    dist_range = feat_range['x'][1]
    speed_range = feat_range['vx'][1]

    for i,row in enumerate(adv_obs):
        if i > 0 and self.v_in_obs[i] is not None:
            self.v_in_obs[i].position[0] = row[1]*dist_range + self.vehicle.position[0]
            self.v_in_obs[i].speed = np.clip( (self.v_in_obs[0].speed+row[3]*speed_range)/self.v_in_obs[i].direction[0], 0, self.v_in_obs[i].MAX_SPEED )

    return self.observation_type.observe()

HighwayEnv.attack_env = attack_env