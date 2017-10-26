#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <memory>

using namespace std;
using namespace cv;

template <typename vehicle_model_t>
class DPClass
{
    using control = typename vehicle_model_t::control;
    using state = typename vehicle_model_t::state;

    // t, p, v, map
    double x_step = 1.0; // time
    double y_step = 1.0; // p
    double z_step = 1.0; // v
    double z_offset = 5.0;
    size_t n_x;
    size_t n_y;
    size_t n_z;

    double horizon_s = 10.0;
    double lookahead_m = 100.0;
    double max_speed = 10.0;


    // control set
    vector<control> U;

    // Matrices to store value function
    vector<Mat> value_img;
    vector<Mat> policy_img;

    std::shared_ptr<vehicle_model_t> car;

public:
    DPClass(std::shared_ptr<vehicle_model_t> model,
            double x_step, double  y_step, double z_step) :
        car(model), x_step(x_step), y_step(y_step), z_step(z_step)
    {
        n_x = horizon_s / x_step; // time
        n_y = lookahead_m / y_step; // position
        n_z = (max_speed + z_offset) / z_step; // velocity
        cout << "Value function table size: " << n_x << "X" << n_y << "X" << n_z << endl;
        for(size_t i = 0; i < n_x; ++i)
        {
            value_img.push_back(Mat(n_y, n_z, CV_64FC1, 1e6));
            policy_img.push_back(Mat(n_y, n_z, CV_64FC1, 0.0));
        }
        // discretize action
        for(double u = -4.0; u < 4.0; u += 1.0)
            U.push_back({u});
    }


    tuple<size_t, size_t, size_t> get_index(const state & x, double time)
    {
        size_t x_index = max<size_t>(0, min<size_t>(time / x_step, n_x - 1));
        size_t y_index = max<size_t>(0, min<size_t>(x[0] / y_step, n_y - 1));
        size_t z_index = max<size_t>(0, min<size_t>((x[1] + z_offset) / z_step, n_z - 1));
        return make_tuple(x_index, y_index, z_index);
    }

    double & value_at(const state & x, double time)
    {
        // get value referece
        size_t x_index, y_index, z_index;
        tie(x_index, y_index, z_index) = get_index(x, time);
        return value_img.at(x_index).at<double>(y_index, z_index);
    }

    control policy_get(const state & x, double time)
    {
        return control{interp(x, time, policy_img)};
        // //
        // get value referece
        // size_t x_index, y_index, z_index;
        // tie(x_index, y_index, z_index) = get_index(x, time);
        // // interpolation here
        // return {policy_img.at(x_index).at<double>(y_index, z_index)};
    }

    void policy_set(const control & u, const state & x, double time)
    {
        // get value referece
        size_t x_index, y_index, z_index;
        tie(x_index, y_index, z_index) = get_index(x, time);
        policy_img.at(x_index).at<double>(y_index, z_index) = u[0];
    }

    double interp(const state & x, double time, const vector<Mat> & img)
    {
        double alpha = (x[0] / y_step);
        alpha -= (int) alpha;
        double beta = ((x[1] + z_offset) / z_step);
        beta -= (int) beta;

        size_t x_index, y_index, z_index;
        tie(x_index, y_index, z_index) = get_index(x, time);

        size_t y_index_plus = min(y_index + 1, n_y -1);
        size_t z_index_plus = min(z_index + 1, n_z);

        auto v1 = (1 - alpha) * img.at(x_index).at<double>(y_index, z_index) +
            alpha * img.at(x_index).at<double>(y_index_plus, z_index);
        auto v2 = (1 - alpha) * img.at(x_index).at<double>(y_index, z_index_plus) +
            alpha * img.at(x_index).at<double>(y_index_plus, z_index_plus);
        return (1 - beta) * v1 + beta * v2;
    }

    double value_get_interp(const state & x, double time)
    {
        if(time >= horizon_s)
            return 0.0;

        return interp(x, time, value_img);
    }

    double value_get(const state & x, double time)
    {
        if(time >= horizon_s)
        {
            return 0.0;
        }
        return value_at(x, time);
    }

    const Mat & value_get(double time)
    {
        size_t x_index = min<size_t>(time / x_step, n_x - 1);
        return value_img.at(x_index);
    }

    double update(const state & x, double time)
    {
        auto & value_ = value_at(x, time);
        for(auto u: U)
        {
            auto x_next = car->dynamics(x, u, x_step);
            auto q = car->cost(x, u, time) * x_step + value_get_interp(x_next, time + x_step);
            if ( q < value_)
            {
                value_ = q;
                policy_set(u, x, time);
            }
        }

        return value_;
    }

    Mat get_all_value()
    {
        Mat V;
        hconcat(value_img, V);
        return V;
    }

    Mat get_all_policy()
    {
        Mat V;
        hconcat(policy_img, V);
        return V;
    }
};
