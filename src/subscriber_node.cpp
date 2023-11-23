#include <ros/ros.h>
#include <cstddef>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <onnxruntime_cxx_api.h>
#include <std_msgs/String.h>
#include <vector>

// using namespace std;

const int class_num = 10;
const int input_height = 28;
const int input_width = 28;
const int channel = 1;
const int batch_size =1;

int classification_result;
ros::Publisher classification_pub;
std::array<float,input_height*input_width*channel*batch_size> input;
const char* input_names[] = {"input.1"};
const char* output_names[] = {"26"};
std::array<int64_t,4>input_shape{batch_size,channel,input_height,input_width};
std::array<float,batch_size*class_num> output;
std::array<int64_t,2> output_shape{batch_size,class_num};

Ort::Value input_tensor{nullptr};
Ort::Value output_tensor{nullptr};


Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Runtime");
Ort::SessionOptions session_options;
Ort::Session session{env, "/home/lhc/catkin_ws/src/detect_pkg/src/model.onnx", session_options};

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
    // cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
    cv::Mat gray_img;
    cv::cvtColor(cv_bridge::toCvShare(msg, "bgr8")->image, gray_img, cv::COLOR_BGR2GRAY);
    // cv::imshow("test",gray_img);
    cv::Mat resized_img;
    cv::resize(gray_img, resized_img, cv::Size(28, 28));
    // cv::imshow("test",resized_img);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, input.data(), input.size(),
          input_shape.data(), input_shape.size());

    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
          memory_info, output.data(), output.size(),
          output_shape.data(), output_shape.size());

    float* input_=input.data();
    for (int i = 0; i < input_height; i++)
    {
        for (int j = 0; j < input_width; j++)
        {
            float tmp=resized_img.ptr<uchar>(i)[j];
            input_[i*input_width+j]=tmp;
        }
    }

    session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, &output_tensor,1);

    classification_result = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

    std_msgs::String classfication_msg;
    classfication_msg.data = std::to_string(classification_result);
    classification_pub.publish(classfication_msg);

    ROS_INFO("classification_result is %s",classfication_msg.data.c_str());
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "img_viewer");
  ros::NodeHandle nh;
  // cv::namedWindow("view");
  cv::startWindowThread();

  session_options.SetIntraOpNumThreads(1);

  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("camera/image", 10, imageCallback);

  classification_pub = nh.advertise<std_msgs::String>("mnist_classification", 10);
  ros::spin();
  // cv::destroyWindow("view");
  return 0;
}