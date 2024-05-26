#include <gazebo/gazebo.hh>
#include <gazebo/common/common.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/rendering/rendering.hh>
#include <gazebo/common/Time.hh>
#include <std_srvs/Empty.h>
#include <ros/ros.h>
#include <gazebo/transport/transport.hh>

using namespace gazebo;

class FloorVisualPlugin : public VisualPlugin
{
public:
    FloorVisualPlugin()
    {
        printf("Visual Plugin Created!\n");
    }
    virtual ~FloorVisualPlugin() {}

protected:
    void Load(rendering::VisualPtr _visual, sdf::ElementPtr _sdf)
    {
        transport::NodePtr node(new transport::Node());
        node->Init("Floor Model");
        this->visual = _visual;

        // ROS Subscriber
        std::string topicName = "/floor_reset"; 
        this->rosSubscriber = node->Subscribe(topicName, this->OnUpdate);
    }

public:
    void OnUpdate(const std_msgs::StringConstPtr &msg)
    {
        const ignition::math::Color ambient = *(new ignition::math::Color(0.7, 0.5, 0.3, 1.0));
        const ignition::math::Color diffuse = *(new ignition::math::Color(0.7, 0.5, 0.3, 1.0));
        const ignition::math::Color specular = *(new ignition::math::Color(0.2, 0.2, 0.2, 1.0));

        this->visual->SetAmbient(ambient);
        this->visual->SetDiffuse(diffuse);
        this->visual->SetSpecular(specular);

        std::cout << "Colors Updated!!" << std::endl;
    }

private: gazebo::rendering::VisualPtr visual;

private: gazebo::transport::SubscriberPtr rosSubscriber
};

GZ_REGISTER_VISUAL_PLUGIN(FloorVisualPlugin)