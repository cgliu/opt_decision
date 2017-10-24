#pragma once

#include <algorithm>
#include <array>
#include <vector>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class pt_map
{
    Mat value_;
    double horizon_s_;
    double lookahead_m_;
    double scale_x_, scale_y_;
    unsigned size_x_, size_y_;

public:
    // width  :
    // height : height in the real world
    pt_map(const Mat & value, double horizon_s, double lookahead_m)
    {
        value_ = value;
        horizon_s_ = horizon_s;
        lookahead_m_ = lookahead_m;
        size_x_ = value.cols;
        size_y_ = value.rows;
        scale_x_ = size_x_ / horizon_s;
        scale_y_ = size_y_ / lookahead_m;
        cout << value_.size() << endl;
        cout << value_.channels() << endl;
    }
    void print_info()
    {
        cout << "map size: " << value_.size() << endl;
    }

    double get(double time, double position)
    {
        // The origin of the image is on the top left,
        // x to the right; y to the bottom
        // x is the column, y is the row
        auto x_index = min<int>(time * scale_x_, size_x_ -1);
        auto y_index = max<int>(size_y_ - 1 - position * scale_y_, 0);
        return value_.at<double>(y_index, x_index) > 100? 0.0l: 1.0l;
    }
};

class vehicle_model{
    double max_speed = 10l;
    double min_speed = 0l;
    pt_map & map;

public:
    using state = array<double, 2>;
    using control = array<double, 1>;

    void print_map_info()
    {
        map.print_info();
    }

    vehicle_model(pt_map & input_map) : map(input_map) {
    }

    state dynamics(state x, control u, double dt)
    {
        x[0] += x[1] * dt + 0.5 * u[0] * dt * dt;
        x[1] += u[0] * dt;
        return x;
    }

    double collision(const state & x, double time)
    {
        return map.get(time, x[0]);
    }

    double cost(const state & x, const control & u, double time)
    {
        double total_cost = 0;

        // collision cost
        total_cost += 100.0 * collision(x, time);

        // odom reward
        total_cost += max_speed - x[1];

        // control effort
        total_cost += 100.0 * u[0] * u[0];

        // speed limits
        double violation = max<double>(0, x[1] - max_speed);
        total_cost += violation * violation;
        violation = max<double>(min_speed - x[1], 0.0);
        total_cost += violation * violation;

        return total_cost;
    }
};
