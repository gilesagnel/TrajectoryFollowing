// gazebo_trajectory_marker_plugin.cpp

#include <gazebo/gazebo.hh>
#include <gazebo/rendering/Visual.hh>
#include <gazebo/rendering/Scene.hh>
#include <gazebo/rendering/RenderEngine.hh>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

namespace gazebo
{
    class TrajectoryMarkerPlugin : public VisualPlugin
    {
    public:
        TrajectoryMarkerPlugin() : VisualPlugin()
        {
            printf("TrajectoryMarkerPlugin Init called\n");
        }

        void Load(rendering::VisualPtr _parent, sdf::ElementPtr /*_sdf*/)
        {
            if (!ros::isInitialized())
            {
                ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
                                 << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
                return;
            }
            ROS_INFO("TrajectoryMarkerPlugin Init called");

            // Store the visual pointer
            // this->visual = _parent;

            // Create a ROS node handle
            // this->rosNode.reset(new ros::NodeHandle());

            // Subscribe to the marker topic
            // this->markerSub = this->rosNode->subscribe("/trajectory_marker", 10, &TrajectoryMarkerPlugin::MarkerCallback, this);
        }

        // Callback function for receiving marker messages
        // void MarkerCallback(const visualization_msgs::MarkerConstPtr &markerMsg)
        // {
        //     // Clear previous trajectory
        //     this->Reset();

        //     // Create a new line strip using make_shared for proper memory management
        //     rendering::DynamicLinesPtr lineStrip =
        //         boost::make_shared<rendering::DynamicLines>(gazebo::rendering::RENDERING_LINE_STRIP);

        //     // Set line properties
        //     lineStrip->setMaterial("Gazebo/Blue");
        //     lineStrip->setPointSize(3.0);

        //     // Add points to the line strip
        //     for (const auto &point : msg->points)
        //     {
        //         ignition::math::Vector3d position(point.x, point.y, point.z);
        //         lineStrip->AddPoint(position);
        //     }

        //     // Add the line strip to the visual
        //     visual->InsertMesh(lineStrip);
        // }

    private:
        // Pointer to the Gazebo visual
        rendering::VisualPtr visual;

        // ROS node handle
        std::unique_ptr<ros::NodeHandle> rosNode;

        // ROS subscriber for marker messages
        ros::Subscriber markerSub;
    };

    GZ_REGISTER_VISUAL_PLUGIN(TrajectoryMarkerPlugin)
} // namespace gazebo
