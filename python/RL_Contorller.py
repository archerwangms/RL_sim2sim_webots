from WebotsInterface import webot_interface,RL_controller
import torch 

if __name__ == "__main__":
    A1_sim = webot_interface()
    controller = RL_controller()
    controller.load("./24_7_20_1500.pt")
    controller.get_inference_policy()
    A1_sim.init_varable()

    while(1):
        controller.step(A1_sim)
        # A1_sim.get_observersion()
        # A1_sim.send_torque(-10.0*A1_sim.joint_vlo)
        # A1_sim.a1_supervisor_class.step(A1_sim.sim_time_step)
        # print(a1_supervisor_node.getDef())

        # zero_torque = torch.zeros(12,dtype=torch.float)
        # A1_sim.send_torque(zero_torque)

