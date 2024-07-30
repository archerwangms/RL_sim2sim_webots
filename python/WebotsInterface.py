from controller import robot,supervisor,motor,position_sensor,inertial_unit,Gyro,keyboard
import numpy as np
import torch 
import torch.nn as nn
from torch.distributions import Normal

class robot_config:
    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
    class action_scales:
        action_scale = 0.25#0.25
    class gait:
        gait_full_period = 0.6       
        stand_scale      = 0.5   
    class normalization:
        clip_observations = 100.
        clip_actions = 100.
   
class webot_interface:
    def __init__(self) -> None:
        self.sim_time_step = 1
        self.supervisor_name = "A1"
        self.a1_joint_name = [['FL_hip_joint','FL_thigh_joint','FL_calf_joint'],
                              ['FR_hip_joint','FR_thigh_joint','FR_calf_joint'],
                              ['RL_hip_joint','RL_thigh_joint','RL_calf_joint'],
                              ['RR_hip_joint','RR_thigh_joint','RR_calf_joint']]
        self.a1_joint_sensor_name =  [['FL_hip_joint_sensor','FL_thigh_joint_sensor','FL_calf_joint_sensor'],
                                      ['FR_hip_joint_sensor','FR_thigh_joint_sensor','FR_calf_joint_sensor'],
                                      ['RL_hip_joint_sensor','RL_thigh_joint_sensor','RL_calf_joint_sensor'],
                                      ['RR_hip_joint_sensor','RR_thigh_joint_sensor','RR_calf_joint_sensor']]
        # default_dof_pos = [ 0.1,0.8,-1.5,
        #                    -0.1,0.8,-1.5,
        #                     0.1,1.,-1.5,
        #                    -0.1,1.,-1.5] #上课a1正常参数
        default_dof_pos = [ 0.1,0.7854,-1.5708,
                           -0.1,0.7854,-1.5708,
                            0.1,0.7854,-1.5708,
                           -0.1,0.7854,-1.5708] # 1500pt 踏步奖励优化版
        torque_limits_list = [20., 55., 55.,
                              20., 55., 55.,
                              20., 55., 55.,
                              20., 55., 55.] #上课a1正常参数 && 1500pt 踏步奖励优化版
        # torque_limits_list = [80., 150., 180.,
        #                       80., 150., 180.,
        #                       80., 150., 180.,
        #                       80., 150., 180.] #a1 heavy参数
        vlo_command = [0.,0.,0.,0.]
        p = 30.#20
        d = 1.5#0.5
        self.joint_node=[]
        self.position_sensor_node=[]

        self.base_lin_vel =  torch.zeros(3,dtype=torch.float)
        self.base_ang_vel = torch.zeros(3,dtype=torch.float)
        self.base_quat = torch.tensor([0,0,0,1],dtype=torch.float) #x y z w
        self.normal_gravity = torch.tensor([0,0,-1],dtype=torch.float)
        self.projected_gravity = torch.tensor([0,0,-1],dtype=torch.float)    
        self.default_dof_pos = torch.tensor(default_dof_pos,dtype=torch.float)
        self.joint_pos = torch.zeros(12,dtype=torch.float)
        self.last_joint_pos = torch.zeros(12,dtype=torch.float)
        self.joint_vlo = torch.zeros(12,dtype=torch.float)
        self.commands = torch.tensor(vlo_command,dtype=torch.float)
        self.KeyCmd = 0.0
        self.LastKeyCmd = 0.0
        self.actions =  torch.zeros(12,dtype=torch.float)
        self.actions_scaled =  torch.zeros(12,dtype=torch.float)
        self.torque_limits = torch.tensor(torque_limits_list,dtype=torch.float)
        self.p_gains = p*torch.ones(12,dtype=torch.float)
        self.d_gains = d*torch.ones(12,dtype=torch.float)
        self.gait_contact_flag = torch.zeros(4,dtype=torch.float)
        self.gait_phase = torch.zeros(1,dtype=torch.float)
        self.gait_contact_flag_leg14 = 1.0 # 1.0*(self.gait_phase > (self.cfg.rewards.gait_full_period*(1.0-self.cfg.rewards.stand_scale))).unsqueeze(dim=1)
        self.gait_contact_flag_leg23 = 0.0 # 1.0*(self.gait_phase < (self.cfg.rewards.gait_full_period*self.cfg.rewards.stand_scale)).unsqueeze(dim=1)
           
        self.a1_supervisor_class = supervisor.Supervisor() ## 实例化supervisor类
        self.a1_supervisor_node = self.a1_supervisor_class.getFromDef(self.supervisor_name)
        if(self.a1_supervisor_node.getDef() != 'A1'):
            print("webots don't have robot named A1 \n")
            print(f"node name : ",self.a1_supervisor_node.getDef())
# motor and sensor node        
        self.np_joint_name = np.array(self.a1_joint_name)        # print(np_joint_name)        
        for leg_num in range(self.np_joint_name.shape[0]):
            for joint_num in range(self.np_joint_name.shape[1]):
                self.joint_node.append(self.a1_supervisor_class.getDevice(self.a1_joint_name[leg_num][joint_num]))
                self.joint_node[leg_num*3+joint_num].enableTorqueFeedback(self.sim_time_step)
                self.position_sensor_node.append(self.a1_supervisor_class.getDevice(self.a1_joint_sensor_name[leg_num][joint_num]))
                self.position_sensor_node[leg_num*3+joint_num].enable(self.sim_time_step)

        self.a1_trans_field = self.a1_supervisor_node.getField("translation")

        self.imu = self.a1_supervisor_class.getDevice("IMU")
        if(self.imu == None):
            print("don't find the imu")        
        else:
            self.imu.enable(self.sim_time_step)

        self.gyro = self.a1_supervisor_class.getDevice("gyro")
        if(self.gyro == None):
            print("don't find the gyro")        
        else:
            self.gyro.enable(self.sim_time_step) 

        self.keyboard = self.a1_supervisor_class.getKeyboard()
        if(self.keyboard == None):
            print("don't find the KEYBOARD")        
        else:
            self.keyboard.enable(self.sim_time_step)
            print("find the keyboard")              

    def get_keyboard_input(self):
        KeyCmd = self.keyboard.get_key()
        if(KeyCmd == 65535):
            isPressed = 0
        else:
            isPressed = 1
        # print("is pressed: ",isPressed)
        # print("KeyCmd: %d"%(KeyCmd))
        
        if(self.LastKeyCmd == KeyCmd and isPressed == 1):
            pass
        else:
            if(KeyCmd == 87):
                self.commands[0]+=0.1 #w
                print("a is pressed")
            if(KeyCmd == 83):
                self.commands[0]-=0.1 #s
            if(KeyCmd == 65):
                self.commands[1]+=0.1 #a
            if(KeyCmd == 68):
                self.commands[1]-=0.1 #d
        # self.commands = torch.clip(self.commands,-0.3,0.3)
        self.commands = torch.clip(self.commands,-0.5,0.5)
        self.LastKeyCmd = KeyCmd

    def get_observersion(self):
        self.get_joint_pos(self.joint_pos)
        self.get_base_quat(self.base_quat)
        self.get_projected_gravity(self.projected_gravity)
        self.update_base_linvel_angvel()
        self.compute_joint_vlo(self.joint_pos,self.last_joint_pos)
        self.last_joint_pos[:] = self.joint_pos[:]
    
    def send_torque(self,torque: torch.tensor):
        for leg_num in range(self.np_joint_name.shape[0]):
            for joint_num in range(self.np_joint_name.shape[1]):
                self.joint_node[leg_num*3+joint_num].setTorque(torque[leg_num*3+joint_num])

    def init_varable(self):
        self.a1_supervisor_class.step()
        self.get_joint_pos(self.joint_pos)
        self.get_joint_pos(self.last_joint_pos)
        self.compute_joint_vlo(self.joint_pos,self.last_joint_pos)

    def compute_joint_vlo(self,pos_now: torch.tensor,
                          pos_before: torch.tensor):
        # vlo_tensor = (pos_now - pos_before)/(1.0/1000.0)
        vlo_tensor = (pos_now - pos_before)/(10.0/1000.0)
        self.joint_vlo[:] = vlo_tensor[:]

    def get_joint_pos(self,pos_tensor: torch.tensor):
        for leg_num in range(self.np_joint_name.shape[0]):
            for joint_num in range(self.np_joint_name.shape[1]):
                pos_tensor[leg_num*3+joint_num] = self.position_sensor_node[leg_num*3+joint_num].getValue()

    def get_base_quat(self,base_quat):
        imu_quat = self.imu.getQuaternion()
        for i in range(4):
            base_quat[i] = imu_quat[i]
    
    def get_projected_gravity(self,projected_gravity):
        projected_gravity_local = self.quat_rotate_inverse(self.base_quat.unsqueeze(0),self.normal_gravity.unsqueeze(0))
        projected_gravity[:] = projected_gravity_local.squeeze(0)

    def quat_rotate_inverse(self,q,v):
        shape = q.shape
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * \
            torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
                shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c

    def compute_torque(self,action: torch.tensor,action_scale):
        actions_scaled = action * action_scale
        self.actions_scaled[:] = actions_scaled[:]
        # print(self.actions_scaled.device)
        command_torque = self.p_gains*(self.actions_scaled + self.default_dof_pos - self.joint_pos) - self.d_gains*self.joint_vlo
        return torch.clip(command_torque, -self.torque_limits, self.torque_limits)
    
    def update_gait_phase(self,gait: robot_config.gait):
        self.gait_phase += 10. * self.sim_time_step/1000.0 # self.decimation
        if self.gait_phase > gait.gait_full_period:
            self.gait_phase[0] = 0.0
        self.gait_contact_flag_leg14 =  1.0*(self.gait_phase > (gait.gait_full_period*(1.0-gait.stand_scale)))
        self.gait_contact_flag_leg23 =  1.0*(self.gait_phase < (gait.gait_full_period*gait.stand_scale))
        # self.gait_contact_flag[0] = self.gait_contact_flag_leg23
        # self.gait_contact_flag[1] = self.gait_contact_flag_leg14
        # self.gait_contact_flag[2] = self.gait_contact_flag_leg14
        # self.gait_contact_flag[3] = self.gait_contact_flag_leg23
        self.gait_contact_flag[:] = torch.cat((self.gait_contact_flag_leg23,self.gait_contact_flag_leg14,
                               self.gait_contact_flag_leg14,self.gait_contact_flag_leg23),
                               dim = 0)
        # print(self.gait_contact_flag)

    def update_base_linvel_angvel(self):
        base_lin_vel_world = torch.zeros_like(self.base_lin_vel)
        base_ang_vel_world = torch.zeros_like(self.base_lin_vel)
        lineVel = self.a1_supervisor_node.getVelocity() # world
        AngVel = self.gyro.getValues() #local
        for i in range(3):
            base_lin_vel_world[i] = lineVel[i]
            self.base_ang_vel[i] = AngVel[i]
        self.base_lin_vel[:] = (self.quat_rotate_inverse(self.base_quat.unsqueeze(0), 
                                                         base_lin_vel_world.unsqueeze(0))).squeeze(0)

class RL_controller(nn.Module):
    def __init__(self,num_nn_input=53,num_actions=12,actor_hidden_dims=[512,256,128],activation="elu",
                 init_noise_std=1.0):
        super().__init__()

        self.obs_scale = robot_config.obs_scales
        self.action_scales = robot_config.action_scales
        self.gait_config = robot_config.gait
        self.normalization = robot_config.normalization
        self.gpu = torch.device("cuda:0")    
        self.decimation = 10   
        self.obs_buf = torch.zeros(num_nn_input,dtype=torch.float)    
        self.commands_scale = torch.tensor([self.obs_scale.lin_vel, self.obs_scale.lin_vel, 
                                            self.obs_scale.ang_vel], requires_grad=False,)

        activation = get_activation(activation)
        mlp_input_dim_a = num_nn_input
        mlp_input_dim_c = num_actions 
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)        
        print(f"Actor MLP: {self.actor}")
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def Compute_obs(self,interface:webot_interface):
        obs_buf = torch.cat(( interface.base_lin_vel * self.obs_scale.lin_vel,
                            interface.base_ang_vel  * self.obs_scale.ang_vel,
                            interface.projected_gravity,
                            interface.commands[:3] * self.commands_scale,
                            (interface.joint_pos - interface.default_dof_pos) * self.obs_scale.dof_pos,
                            interface.joint_vlo * self.obs_scale.dof_vel,
                            interface.actions,
                            interface.gait_contact_flag,
                            interface.gait_phase    
                            ),dim=-1)
        self.obs_buf = obs_buf.to(self.gpu)
        clip_obs = self.normalization.clip_observations
        return torch.clip(self.obs_buf, -clip_obs, clip_obs)
    
    def load(self,path):
        loaded_dict = torch.load(path)
        pretrained_model = loaded_dict["model_state_dict"]
        state_dirt = {k:v for k,v in pretrained_model.items() if ("actor" in k or "std" in k)}
        self.load_state_dict(state_dirt)

    def get_inference_policy(self):
        self.eval()
        self.to(self.gpu)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean
    
    def step(self,interface:webot_interface):
        for i in range(self.decimation):
            interface.get_keyboard_input()
            interface.get_observersion()
            if i == 0 :
                interface.update_gait_phase(self.gait_config)
                obs = self.Compute_obs(interface)
                actions = self.act_inference(obs.detach())
                clip_actions = self.normalization.clip_actions
                actions = torch.clip(actions, -clip_actions, clip_actions)
                interface.actions[:] = actions[:]
            wb_torque = interface.compute_torque(interface.actions,self.action_scales.action_scale)
            interface.send_torque(wb_torque)
            interface.a1_supervisor_class.step(interface.sim_time_step)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

