#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

class pt_map
{
    Mat value_;
    double horizon_s_;
    double lookahead_m_;
    double scale_x_, scale_y_;
    size_t size_x_, size_y_;

public:
    pt_map(const Mat & value, double horizon_s, double lookahead_m)
    {
        value_ = value;
        horizon_s_ = horizon_s;
        lookahead_m_ = lookahead_m;
        size_x_ = value.cols;
        size_y_ = value.rows;

        scale_x_ = size_x_ / horizon_s;
        scale_y_ = size_y_ / lookahead_m;

        // display
        cout << value_.size() << endl;
        cout << value_.channels() << endl;
    }

    string show_info()
    {
        cout << "map size: " <<
            size_x_ << "X" << size_y_ << endl;
    }

    Mat & get_value()
    {
        return value_;
    }

    tuple<size_t, size_t> get_index(double position, double time)
    {
        auto x_index = min<size_t>(max<size_t>(0, time * scale_x_), size_x_ -1);
        auto y_index = min<size_t>(max<size_t>(0, position * scale_y_), size_y_ - 1);
        return make_tuple(x_index, y_index);
    }

    double & at(double position, double time)
    {
        size_t x_index, y_index;
        tie(x_index, y_index) = get_index(position, time);
        return value_.at<double>(y_index, x_index);
    }

    double get(double position, double time)
    {
        return at(position, time);
    }

    void draw_circle(double position, double time)
    {
        size_t x_index, y_index;
        tie(x_index, y_index) = get_index(position, time);
        circle(value_, {(int)x_index, (int)y_index}, 5, (255, 0, 0), -1);
    }

};
