// Graymatics_v3.cpp

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>

int main()
{
    std::cout << "Running!\n";

    int64_t input_size = 640;
    int64_t num_Channels = 3;
    int64_t batch_size = 1;
    int64_t num_InputElements = num_Channels * input_size * input_size;
    int64_t num_Classes = 80; //Yolov5 pre-build have 80  

    float threshold_Score = 0.5;
    float threshold_NMS = 0.45;
    float threshold_Confidence = 0.045;


    // Set up ONNX runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Inference");
    Ort::SessionOptions session_options;


    //const wchar_t* model_path = L"D:/Daniel_Beta/AI/Graymatics_v3/Graymatics_v3/assets/yolov5n.onnx";
    const wchar_t* model_path = L"assets/yolov5n.onnx";

    std::vector<std::string> labels = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
    };

    //std::string video_path = "D:/Daniel_Beta/AI/Graymatics_v3/Graymatics_v3/assets/ML_video_test.mp4";
    std::string video_path = "assets/ML_video_test.mp4";

    


    Ort::Session session(env, model_path, session_options);

    // Setup video and get some video info 
    cv::VideoCapture cap(video_path);//cv::VideoCapture cap(0); //webcam    
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open video file\n";
        return -1;
    }

    //get the frames per seconds of the video
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "fps: " << fps;
    // Get video frame size
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Frame width: " << frame_width;
    std::cout << "Frame height: " << frame_height;

    
    //create a window
    cv::namedWindow("Video_window", cv::WINDOW_AUTOSIZE); 


    //Video object to create the infered video  
    cv::VideoWriter output_video("assets/ML_video_test_output_mp4.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), fps, cv::Size(frame_width, frame_height));
    
    
                          
    //Pre-Process and Process each frame from the video with the model
    cv::Mat frame;
    while (true)
    {
        //Get the next frame from the video
        cap >> frame;
        //frame = cv::imread("assets/zidane.jpg");

        //If frame is empty that means we finished the video
        if (frame.empty()) break;

        //If Need to rotate video 
        /*cv::transpose(frame, frame);
        cv::flip(frame, frame, 0);*/

        //Resizing frame 
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(input_size, input_size));

        //Converting BRG to RGB colour
        cv::Mat rgb_frame;
        cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);


        //Converting frame to float and normalising (0-1), 
        rgb_frame.convertTo(rgb_frame, CV_32F, 1.0 / 255.0);

        //Reshaping frame to a single row so we can use for input model 
        cv::Mat reshaped_frame = rgb_frame.reshape(1, 1);

        //Converting frame to a vector of floats
        std::vector<float> frame_data(reshaped_frame.begin<float>(), reshaped_frame.end<float>());

        //Creating tensor for model input
        int64_t input_tensor_size = input_size * input_size * num_Channels; //640*640*3 == num_InoutElements 
        std::vector<int64_t> input_tensor_shape = { batch_size, num_Channels, input_size, input_size }; //1,3,640,640


        //Specifying that the tensore allocated on the CPU
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        //create tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, frame_data.data(), num_InputElements, input_tensor_shape.data(), input_tensor_shape.size());

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr input_Name = session.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr output_Name = session.GetOutputNameAllocated(0, allocator);
        const std::array<const char*, 1> input_Names = { input_Name.get() };
        const std::array<const char*, 1> output_Names = { output_Name.get() };
        input_Name.release();
        output_Name.release();

        // Run the model
        std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_Names.data(), &input_tensor, 1, output_Names.data(), 1);


        Ort::Value& output_tensor = output_tensors.front();
        float* float_Arr = output_tensor.GetTensorMutableData<float>();

        /* Get tensor shape ==> result seem to fit YOLOv5 model's output
        * GetElementType: 1
        * Dimensions of the output: 3
        * Shape of the output: 1, 25200, 85,
        */
        auto output_Info = output_tensor.GetTensorTypeAndShapeInfo();
        auto Shape_Info = output_Info.GetShape();
        /*std::cout << "GetElementType: " << output_Info.GetElementType() << "\n";
        std::cout << "Dimensions of the output: " << output_Info.GetShape().size() << "\n";
        std::cout << "Shape of the output: ";
        for (unsigned int shapeI = 0; shapeI < output_Info.GetShape().size(); shapeI++){std::cout << output_Info.GetShape()[shapeI] << ", ";}*/
        std::vector<std::vector<std::vector<float>>> output_boxes(Shape_Info[0], std::vector<std::vector<float>>(Shape_Info[1], std::vector<float>(Shape_Info[2])));

        for (int i = 0; i < Shape_Info[0]; i++) { //1
            for (int j = 0; j < Shape_Info[1]; j++) { //25200
                for (int k = 0; k < Shape_Info[2]; k++) { //85
                    output_boxes[i][j][k] = *float_Arr;
                    float_Arr++;
                }
            }
        }

        //// Iterate over all boxes for the first frame in the batch (in our context there is only one frame
        //for (int box_index = 0; box_index < output_boxes[0].size(); ++box_index) {
        //    //Get objectness score/confidence
        //    float objectness_Score = output_boxes[0][box_index][4]; 
        //    //Get class probabilities
        //    std::vector<float> class_scores(output_boxes[0][box_index].begin() + 5, output_boxes[0][box_index].end());
        //    //Get index of class with the highest score
        //    int class_index = std::distance(class_scores.begin(), std::max_element(class_scores.begin(), class_scores.end()));
        //    //Get max class score
        //    float max_class_score = *std::max_element(class_scores.begin(), class_scores.end());
        //    if (objectness_Score > threshold_Confidence) { 
        //        std::cout << "!!! Detected class " << labels[class_index] << " with score " << objectness_Score << "\n";
        //    }
        //    //else {
        //    //    //If objectness score is below threshold, just print max class score
        //    //    std::cout << "--- Max class score: " << max_class_score << " for class " << labels[class_index] << "\n";
        //    //}
        //}


        //Drawing the bounding boxes and creating new video with them
        //SCaling the bounding box coordinates from the resized image to the original image
        //float scale = static_cast<float>(frame_height) / input_size;
        //
        for (int box_index = 0; box_index < output_boxes[0].size(); ++box_index) {

            //Get objectness score/confidence
            float objectness_Score = output_boxes[0][box_index][4];

            if (objectness_Score > threshold_Confidence) {
                //Get class probabilities
                std::vector<float> class_scores(output_boxes[0][box_index].begin() + 5, output_boxes[0][box_index].end());
                //Get index of class with the highest score
                int class_index = std::distance(class_scores.begin(), std::max_element(class_scores.begin(), class_scores.end()));
                //Get max class score
                float max_class_score = *std::max_element(class_scores.begin(), class_scores.end());

                std::cout << "!!! Detected class " << labels[class_index] << " with score " << objectness_Score << "\n";

                // Get bounding box coordinates
                int x_center = static_cast<int>(output_boxes[0][box_index][0]);// *input_size); // *frame_width);
                int y_center = static_cast<int>(output_boxes[0][box_index][1]);//* input_size); // * frame_height);
                int box_width = static_cast<int>(output_boxes[0][box_index][2] * 0.8);//* input_size); // * frame_width);
                int box_height = static_cast<int>(output_boxes[0][box_index][3] * 0.8);//* input_size); // * frame_height);

                // Convert to top-left corner coordinates
                int x = x_center - box_width / 2;
                int y = y_center - box_height / 2;

                //Draw bounding box and label on the frame
                cv::rectangle(resized_frame, cv::Point(x, y), cv::Point(x + box_width, y + box_height), cv::Scalar(0, 255, 0), 2);
                cv::putText(resized_frame, labels[class_index], cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
            }
        }

        //Add frame to video object
        output_video.write(resized_frame);

        //Display frame in window
        cv::imshow("Video_window", resized_frame);

        //If any key is pressed exit window
        if (cv::waitKey(1) >= 0) break;

    }

    //
    output_video.release();

    cv::destroyAllWindows();
    std::cout << "Done!\n";
}


