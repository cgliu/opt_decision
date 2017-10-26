#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class pt_map
{
    cv::Mat value_;
    double horizon_s_;
    double lookahead_m_;
    double scale_x_, scale_y_;
    std::size_t size_x_, size_y_;

public:
    pt_map(const cv::Mat & value, double horizon_s, double lookahead_m)
    {
        value_ = value;
        horizon_s_ = horizon_s;
        lookahead_m_ = lookahead_m;
        size_x_ = value.cols;
        size_y_ = value.rows;

        scale_x_ = size_x_ / horizon_s;
        scale_y_ = size_y_ / lookahead_m;

        // display
        std::cout << "Map size in pixel: " << value_.size() << std::endl;
    }

    std::string show_info()
    {
        std::cout << "map size: " <<
            size_x_ << "X" << size_y_ << std::endl;
    }

    cv::Mat & get_value()
    {
        return value_;
    }

    std::tuple<std::size_t, std::size_t> get_index(double position, double time)
    {
        auto x_index = std::min<std::size_t>(std::max<std::size_t>(0, time * scale_x_), size_x_ -1);
        auto y_index = std::min<std::size_t>(std::max<std::size_t>(0, position * scale_y_), size_y_ - 1);
        return std::make_tuple(x_index, y_index);
    }

    double & at(double position, double time)
    {
        std::size_t x_index, y_index;
        std::tie(x_index, y_index) = get_index(position, time);
        return value_.at<double>(y_index, x_index);
    }

    double get(double position, double time)
    {
        return at(position, time);
    }

    void draw_circle(double position, double time)
    {
        std::size_t x_index, y_index;
        std::tie(x_index, y_index) = get_index(position, time);
        cv::circle(value_, {(int)x_index, (int)y_index}, 5, (255, 0, 0), -1);
    }

};
