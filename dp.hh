#pragma once

/*
  A class for Dynamic Programming
 */

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

template <typename vehicle_model_t>
class DPClass
{
    using control = typename vehicle_model_t::control;
    using state = typename vehicle_model_t::state;

    // t, p, v, map
    double dt = 1.0;
    double x_step = 1.0; // time
    double y_step = 1.0; // p
    double z_step = 1.0; // v

    double horizon_s = 10.0;
    double lookahead_m = 100.0;
    double max_speed = 10l;

    unsigned n_x = horizon_s / x_step; // time
    unsigned n_y = lookahead_m / y_step; // position
    unsigned n_z = max_speed / z_step; // velocity

    // Candidate controls
    vector<control> U = {{-2}, {-1}, {0}, {1}, {2}};

    // matrices to store value function
    vector<Mat> value_img;
    vector<Mat> policy_img;

    // matrices to store optimal policy
    // vector<vector<vector<control>>> policy;
    vehicle_model_t * car;

public:
    DPClass(vehicle_model_t * model) : car(model)
    {
        cout << "Value function table is " << n_x << "X" << n_y << "X" << n_z << endl;
        for(size_t i = 0; i < n_x; ++i)
        {
            value_img.push_back(Mat(n_y, n_z, CV_64FC1, 1e5));
            policy_img.push_back(Mat(n_y, n_z, CV_64FC1, 0.0));
        }
        // policy.assign(n_x, vector<vector<control>> (n_y, vector<control> (n_z, {0})));
        car->print_map_info();
    }


    tuple<unsigned, unsigned, unsigned> get_index(const state & x, double time)
    {
        unsigned x_index = min<unsigned>(time / x_step, n_x - 1);
        // for OpenCV, the origin is on the top left
        // x is downwards
        // y is rightwards
        // Revert y-axis to make it look more intuitive
        unsigned y_index = max<unsigned>(0, min<unsigned>(n_y - 1 - (x[0] / y_step), n_y - 1));
        unsigned z_index = min<unsigned>(x[1] / z_step, n_z - 1);
        return make_tuple(x_index, y_index, z_index);
    }

    double & value_at(const state & x, double time)
    {
        // get value referece
        unsigned x_index, y_index, z_index;
        tie(x_index, y_index, z_index) = get_index(x, time);
        return value_img.at(x_index).at<double>(y_index, z_index);
    }

    control policy_get(const state & x, double time)
    {
        // get value referece
        unsigned x_index, y_index, z_index;
        tie(x_index, y_index, z_index) = get_index(x, time);
        if(find(U.begin(), U.end(),
                control({policy_img.at(x_index).at<double>(y_index, z_index)}))
           == U.end())
        {
            cout << x_index << " " << y_index << " "
                 << z_index << " "
                 << policy_img.at(x_index).at<double>(y_index, z_index) << endl;
            cout << "error" << endl;
        }

        return {policy_img.at(x_index).at<double>(y_index, z_index)};
    }

    void policy_set(const control & u, const state & x, double time)
    {
        // get value referece
        unsigned x_index, y_index, z_index;
        tie(x_index, y_index, z_index) = get_index(x, time);
        policy_img.at(x_index).at<double>(y_index, z_index) = u[0];
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
        unsigned x_index = min<unsigned>(time / x_step, n_x - 1);
        return value_img.at(x_index);
    }

    double update(const state & x, double time)
    {
        auto & value_ = value_at(x, time);
        for(auto u: U)
        {
            auto x_next = car->dynamics(x, u, dt);
            auto q = car->cost(x, u, time) + value_get(x_next, time + dt);
            if ( q < value_)
            {
                value_ = q;
                policy_set(control(u), x, time);
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
