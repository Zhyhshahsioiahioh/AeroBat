import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = np.zeros(3)
        # physical velocity
        self.p_vel = np.zeros(3)

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        # physical angle
        self.phi = 0  # 0-2pi
        self.theta = 0
        self.psi = 0
        # physical angular velocity
        self.p_omg = 0
        self.last_a = np.array([0, 0, 0])
        # norm of physical velocity
        self.V = 0
        self.controller = 0

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = np.zeros(3)
        # communication action
        self.c = None

# action of the agent
class LastAction(object):
    def __init__(self):
        # physical action
        self.u = np.zeros(3)
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name
        self.name = ''
        # properties:
        self.size = 1.0
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.max_angular = None
        self.max_accel = None
        self.accel = None
        # state: including internal/mental state p_pos, p_vel
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # commu channel
        self.channel = None

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()
        self.R = None
        self.delta = None
        self.Ls = None
        self.movable = False

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()

        # agents are dummy
        self.dummy = False
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state: including communication state(communication utterance) c and internal/mental state p_pos, p_vel
        self.state = AgentState()
        # action: physical action u & communication action c
        self.action = Action()
        # script behavior to execute
        self.last_action = LastAction
        self.action_callback = None
        self.goal = None
        self.done = False
        self.policy_action = np.zeros(3)
        self.network_action = np.zeros(3)
        self.option = 0

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.my_agents = []
        self.landmarks = []
        self.walls = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 3
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping（阻尼）
        self.damping = 0 # 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None
        self.world_length = 250
        self.world_step = 0
        self.num_agents = 1
        self.num_landmarks = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        """
        cached_dist_vect: 类似图论矩阵, 记录i_a和i_b相对位置关系的向量
        """
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities （size相加?）
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))  # N*N数组，N为智能体个数
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist
            # 实对称距离矩阵

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)

        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)  # bool

    def assign_agent_colors(self):
        """Give each agent a simple default color if none is set."""
        palette = [
            (0.25, 0.25, 0.75),
            (0.75, 0.25, 0.25),
            (0.25, 0.75, 0.25),
        ]
        for idx, agent in enumerate(self.agents):
            if getattr(agent, "color", None) is None:
                agent.color = palette[idx % len(palette)]

    # landmark color
    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

    # update state of the world
    def step(self):
        """
        Advance one simulation step.

        If the attached scenario provides a custom step(world) it is used.
        Otherwise each agent's action_callback(agent, world) is invoked (when
        present) and the resulting actions are integrated.
        """
        if hasattr(self, 'scenario') and hasattr(self.scenario, 'step'):
            return self.scenario.step(self)

        u = [None] * len(self.agents)
        for i, agent in enumerate(self.agents):
            if agent.action_callback is not None:
                agent.action.u = agent.action_callback(agent, self)
            u[i] = agent.action.u

        self.integrate_state(u)
        self.world_step += 1

    # gather agent action forces
    def apply_action_force(self, u):
        # set applied forces
        for i, agent in enumerate(self.agents):
            u[i] = agent.action.u
        return u


    def integrate_state(self, u):  # u:[...[ax, ay, az]...]
        for i, agent in enumerate(self.agents):
            v = agent.state.p_vel + u[i] * self.dt
            speed = np.linalg.norm(v)
            if speed != 0 and agent.max_speed is not None:
                v = v / speed * agent.max_speed

            if getattr(agent, 'done', False):
                agent.state.p_vel = np.zeros(3)
            else:
                v_x, v_y, v_z = v[0], v[1], v[2]
                theta = np.arctan2(v_y, v_x)
                if theta < 0:
                    theta += np.pi * 2
                horizontal_norm = np.sqrt(v_x ** 2 + v_y ** 2)
                elevation = np.arctan2(v_z, horizontal_norm)
                agent.state.phi = theta
                agent.state.theta = elevation
                agent.state.p_vel = np.array([v_x, v_y, v_z])
            agent.state.p_pos += agent.state.p_vel * self.dt

