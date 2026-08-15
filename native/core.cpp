#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

bool to_ints(PyObject* obj, std::vector<long long>& out, const char* error_message) {
    PyObject* seq = PySequence_Fast(obj, error_message);
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    out.reserve(static_cast<std::size_t>(n));
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        const long long value = PyLong_AsLongLong(items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return false;
        }
        out.push_back(value);
    }
    Py_DECREF(seq);
    return true;
}

bool to_doubles(PyObject* obj, std::vector<double>& out, const char* error_message) {
    PyObject* seq = PySequence_Fast(obj, error_message);
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    out.reserve(static_cast<std::size_t>(n));
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        const double value = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return false;
        }
        out.push_back(value);
    }
    Py_DECREF(seq);
    return true;
}

PyObject* version(PyObject*, PyObject*) {
    return PyUnicode_FromString("0.1.1");
}

PyObject* smooth_scores(PyObject*, PyObject* args) {
    PyObject* raw_obj = nullptr;
    double weight_visual = 0.0;
    double weight_temporal = 0.0;
    int pool_window = 0;
    if (!PyArg_ParseTuple(args, "Oddi", &raw_obj, &weight_visual, &weight_temporal, &pool_window)) {
        return nullptr;
    }
    if (pool_window < 0) {
        PyErr_SetString(PyExc_ValueError, "pool_window must be >= 0");
        return nullptr;
    }

    Py_buffer view{};
    if (PyObject_GetBuffer(raw_obj, &view, PyBUF_CONTIG_RO | PyBUF_FORMAT) != 0) {
        PyErr_SetString(PyExc_TypeError, "raw_scores must expose a contiguous float buffer");
        return nullptr;
    }

    const bool is_f32 = view.itemsize == 4 && view.format && std::strchr(view.format, 'f');
    const bool is_f64 = view.itemsize == 8 && view.format && std::strchr(view.format, 'd');
    if (!is_f32 && !is_f64) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "raw_scores must be float32 or float64");
        return nullptr;
    }

    const Py_ssize_t n = view.len / view.itemsize;
    std::vector<float> output(static_cast<std::size_t>(n));

    auto value_at = [&](Py_ssize_t index) -> double {
        if (is_f32) {
            return static_cast<double>(static_cast<const float*>(view.buf)[index]);
        }
        return static_cast<const double*>(view.buf)[index];
    };

    for (Py_ssize_t i = 0; i < n; ++i) {
        const Py_ssize_t start = std::max<Py_ssize_t>(0, i - pool_window);
        const Py_ssize_t end = std::min<Py_ssize_t>(n, i + pool_window + 1);
        double sum = 0.0;
        for (Py_ssize_t j = start; j < end; ++j) sum += value_at(j);
        const double local_mean = sum / static_cast<double>(end - start);
        output[static_cast<std::size_t>(i)] = static_cast<float>(
            weight_visual * value_at(i) + weight_temporal * local_mean
        );
    }

    PyBuffer_Release(&view);
    return PyBytes_FromStringAndSize(
        reinterpret_cast<const char*>(output.data()),
        n * static_cast<Py_ssize_t>(sizeof(float))
    );
}

PyObject* temporal_nms_indices(PyObject*, PyObject* args) {
    PyObject* video_ids_obj = nullptr;
    PyObject* frame_ids_obj = nullptr;
    long long min_gap = 0;
    long long top_k = 0;
    if (!PyArg_ParseTuple(args, "OOLL", &video_ids_obj, &frame_ids_obj, &min_gap, &top_k)) {
        return nullptr;
    }
    if (min_gap < 0 || top_k < 0) {
        PyErr_SetString(PyExc_ValueError, "min_gap and top_k must be >= 0");
        return nullptr;
    }

    PyObject* videos = PySequence_Fast(video_ids_obj, "video_ids must be a sequence");
    if (!videos) return nullptr;
    PyObject* frames = PySequence_Fast(frame_ids_obj, "frame_ids must be a sequence");
    if (!frames) {
        Py_DECREF(videos);
        return nullptr;
    }

    const Py_ssize_t n = PySequence_Fast_GET_SIZE(videos);
    if (PySequence_Fast_GET_SIZE(frames) != n) {
        Py_DECREF(videos);
        Py_DECREF(frames);
        PyErr_SetString(PyExc_ValueError, "video_ids and frame_ids must have the same length");
        return nullptr;
    }

    PyObject* output = PyList_New(0);
    if (!output) {
        Py_DECREF(videos);
        Py_DECREF(frames);
        return nullptr;
    }

    std::unordered_map<std::string, std::vector<long long>> selected_by_video;
    PyObject** video_items = PySequence_Fast_ITEMS(videos);
    PyObject** frame_items = PySequence_Fast_ITEMS(frames);

    for (Py_ssize_t i = 0; i < n && PyList_GET_SIZE(output) < top_k; ++i) {
        const char* video_id = PyUnicode_AsUTF8(video_items[i]);
        if (!video_id) {
            Py_DECREF(output);
            Py_DECREF(videos);
            Py_DECREF(frames);
            return nullptr;
        }
        const long long frame_id = PyLong_AsLongLong(frame_items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(output);
            Py_DECREF(videos);
            Py_DECREF(frames);
            return nullptr;
        }

        auto& chosen = selected_by_video[video_id];
        bool too_close = false;
        for (const long long prior : chosen) {
            if (std::abs(frame_id - prior) < min_gap) {
                too_close = true;
                break;
            }
        }
        if (!too_close) {
            chosen.push_back(frame_id);
            PyObject* index = PyLong_FromSsize_t(i);
            if (!index || PyList_Append(output, index) != 0) {
                Py_XDECREF(index);
                Py_DECREF(output);
                Py_DECREF(videos);
                Py_DECREF(frames);
                return nullptr;
            }
            Py_DECREF(index);
        }
    }

    Py_DECREF(videos);
    Py_DECREF(frames);
    return output;
}

struct Region {
    long long start = 0;
    long long end = 0;
    std::vector<long long> frames;
    double max_score = 0.0;
};

PyObject* merge_temporal_regions(PyObject*, PyObject* args) {
    PyObject* frame_ids_obj = nullptr;
    PyObject* scores_obj = nullptr;
    long long delta_frames = 0;
    long long total_frames = 0;
    long long max_regions = 0;
    if (!PyArg_ParseTuple(args, "OOLLL", &frame_ids_obj, &scores_obj, &delta_frames, &total_frames, &max_regions)) {
        return nullptr;
    }
    if (delta_frames < 0 || total_frames <= 0 || max_regions < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid temporal-region limits");
        return nullptr;
    }

    std::vector<long long> frame_ids;
    std::vector<double> scores;
    if (!to_ints(frame_ids_obj, frame_ids, "frame_ids must be a sequence") ||
        !to_doubles(scores_obj, scores, "scores must be a sequence")) {
        return nullptr;
    }
    if (frame_ids.size() != scores.size()) {
        PyErr_SetString(PyExc_ValueError, "frame_ids and scores must have the same length");
        return nullptr;
    }

    struct Interval {
        long long start;
        long long end;
        long long frame;
        double score;
    };
    std::vector<Interval> intervals;
    intervals.reserve(frame_ids.size());
    for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        const long long frame = frame_ids[i];
        intervals.push_back({
            std::max<long long>(0, frame - delta_frames),
            std::min<long long>(total_frames - 1, frame + delta_frames),
            frame,
            scores[i],
        });
    }
    std::stable_sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
        return a.start < b.start;
    });

    std::vector<Region> merged;
    for (const auto& interval : intervals) {
        if (merged.empty() || interval.start > merged.back().end) {
            Region region;
            region.start = interval.start;
            region.end = interval.end;
            region.frames.push_back(interval.frame);
            region.max_score = interval.score;
            merged.push_back(std::move(region));
        } else {
            auto& region = merged.back();
            region.end = std::max(region.end, interval.end);
            region.frames.push_back(interval.frame);
            region.max_score = std::max(region.max_score, interval.score);
        }
    }

    if (static_cast<long long>(merged.size()) > max_regions) {
        std::stable_sort(merged.begin(), merged.end(), [](const Region& a, const Region& b) {
            return a.max_score > b.max_score;
        });
        merged.resize(static_cast<std::size_t>(max_regions));
        std::stable_sort(merged.begin(), merged.end(), [](const Region& a, const Region& b) {
            return a.start < b.start;
        });
    }

    PyObject* output = PyList_New(static_cast<Py_ssize_t>(merged.size()));
    if (!output) return nullptr;
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(merged.size()); ++i) {
        const auto& region = merged[static_cast<std::size_t>(i)];
        PyObject* frame_list = PyList_New(static_cast<Py_ssize_t>(region.frames.size()));
        if (!frame_list) {
            Py_DECREF(output);
            return nullptr;
        }
        for (Py_ssize_t j = 0; j < static_cast<Py_ssize_t>(region.frames.size()); ++j) {
            PyObject* frame = PyLong_FromLongLong(region.frames[static_cast<std::size_t>(j)]);
            if (!frame) {
                Py_DECREF(frame_list);
                Py_DECREF(output);
                return nullptr;
            }
            PyList_SET_ITEM(frame_list, j, frame);
        }

        PyObject* tuple = PyTuple_New(4);
        if (!tuple) {
            Py_DECREF(frame_list);
            Py_DECREF(output);
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple, 0, PyLong_FromLongLong(region.start));
        PyTuple_SET_ITEM(tuple, 1, PyLong_FromLongLong(region.end));
        PyTuple_SET_ITEM(tuple, 2, frame_list);
        PyTuple_SET_ITEM(tuple, 3, PyFloat_FromDouble(region.max_score));
        PyList_SET_ITEM(output, i, tuple);
    }
    return output;
}

struct State {
    double score = -std::numeric_limits<double>::infinity();
    std::vector<long long> path;
    bool valid = false;
};

bool better(const State& candidate, const State& current) {
    if (!candidate.valid) return false;
    if (!current.valid) return true;
    if (candidate.score > current.score) return true;
    if (candidate.score < current.score) return false;
    return std::lexicographical_compare(
        candidate.path.begin(), candidate.path.end(), current.path.begin(), current.path.end()
    );
}

PyObject* align_trake_events(PyObject*, PyObject* args) {
    PyObject* frames_by_event_obj = nullptr;
    PyObject* scores_by_event_obj = nullptr;
    double transition_penalty = 0.0;
    PyObject* max_gap_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOdO", &frames_by_event_obj, &scores_by_event_obj, &transition_penalty, &max_gap_obj)) {
        return nullptr;
    }

    bool has_max_gap = max_gap_obj != Py_None;
    long long max_gap = 0;
    if (has_max_gap) {
        max_gap = PyLong_AsLongLong(max_gap_obj);
        if (PyErr_Occurred()) return nullptr;
        if (max_gap < 0) {
            PyErr_SetString(PyExc_ValueError, "max_gap must be >= 0 or None");
            return nullptr;
        }
    }

    PyObject* frame_events = PySequence_Fast(frames_by_event_obj, "frame_ids_by_event must be a sequence");
    if (!frame_events) return nullptr;
    PyObject* score_events = PySequence_Fast(scores_by_event_obj, "scores_by_event must be a sequence");
    if (!score_events) {
        Py_DECREF(frame_events);
        return nullptr;
    }

    const Py_ssize_t event_count = PySequence_Fast_GET_SIZE(frame_events);
    if (event_count == 0) {
        Py_DECREF(frame_events);
        Py_DECREF(score_events);
        Py_RETURN_NONE;
    }
    if (PySequence_Fast_GET_SIZE(score_events) != event_count) {
        Py_DECREF(frame_events);
        Py_DECREF(score_events);
        PyErr_SetString(PyExc_ValueError, "event frame/score lists must have matching lengths");
        return nullptr;
    }

    struct Candidate {
        long long frame;
        double score;
    };
    std::vector<std::vector<Candidate>> events;
    events.reserve(static_cast<std::size_t>(event_count));

    PyObject** frame_items = PySequence_Fast_ITEMS(frame_events);
    PyObject** score_items = PySequence_Fast_ITEMS(score_events);
    for (Py_ssize_t event_index = 0; event_index < event_count; ++event_index) {
        std::vector<long long> frames;
        std::vector<double> scores;
        if (!to_ints(frame_items[event_index], frames, "event frame IDs must be a sequence") ||
            !to_doubles(score_items[event_index], scores, "event scores must be a sequence")) {
            Py_DECREF(frame_events);
            Py_DECREF(score_events);
            return nullptr;
        }
        if (frames.size() != scores.size()) {
            Py_DECREF(frame_events);
            Py_DECREF(score_events);
            PyErr_SetString(PyExc_ValueError, "each event must have matching frame and score lengths");
            return nullptr;
        }
        if (frames.empty()) {
            Py_DECREF(frame_events);
            Py_DECREF(score_events);
            Py_RETURN_NONE;
        }

        std::vector<Candidate> event;
        event.reserve(frames.size());
        for (std::size_t i = 0; i < frames.size(); ++i) event.push_back({frames[i], scores[i]});
        std::stable_sort(event.begin(), event.end(), [](const Candidate& a, const Candidate& b) {
            return a.frame < b.frame;
        });
        events.push_back(std::move(event));
    }
    Py_DECREF(frame_events);
    Py_DECREF(score_events);

    std::vector<State> states;
    states.reserve(events.front().size());
    for (const auto& candidate : events.front()) {
        states.push_back({candidate.score, {candidate.frame}, true});
    }

    for (std::size_t event_index = 1; event_index < events.size(); ++event_index) {
        std::vector<State> next_states(events[event_index].size());
        for (std::size_t current_index = 0; current_index < events[event_index].size(); ++current_index) {
            const auto& current = events[event_index][current_index];
            State best;
            for (std::size_t prior_index = 0; prior_index < events[event_index - 1].size(); ++prior_index) {
                if (!states[prior_index].valid) continue;
                const long long gap = current.frame - events[event_index - 1][prior_index].frame;
                if (gap <= 0 || (has_max_gap && gap > max_gap)) continue;

                State proposal;
                proposal.valid = true;
                proposal.score = states[prior_index].score + current.score - transition_penalty * static_cast<double>(gap);
                proposal.path = states[prior_index].path;
                proposal.path.push_back(current.frame);
                if (better(proposal, best)) best = std::move(proposal);
            }
            next_states[current_index] = std::move(best);
        }
        states = std::move(next_states);
    }

    State best;
    for (const auto& state : states) {
        if (state.valid && state.path.size() == events.size() && better(state, best)) best = state;
    }
    if (!best.valid) Py_RETURN_NONE;

    PyObject* path = PyList_New(static_cast<Py_ssize_t>(best.path.size()));
    if (!path) return nullptr;
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(best.path.size()); ++i) {
        PyObject* frame = PyLong_FromLongLong(best.path[static_cast<std::size_t>(i)]);
        if (!frame) {
            Py_DECREF(path);
            return nullptr;
        }
        PyList_SET_ITEM(path, i, frame);
    }

    PyObject* result = PyTuple_New(2);
    if (!result) {
        Py_DECREF(path);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(best.score));
    PyTuple_SET_ITEM(result, 1, path);
    return result;
}

PyMethodDef methods[] = {
    {"version", version, METH_NOARGS, "Return native-core ABI version."},
    {"smooth_scores", smooth_scores, METH_VARARGS, "Smooth temporal scores and return float32 bytes."},
    {"temporal_nms_indices", temporal_nms_indices, METH_VARARGS, "Return stable temporal-NMS indices."},
    {"merge_temporal_regions", merge_temporal_regions, METH_VARARGS, "Merge overlapping temporal candidate windows."},
    {"align_trake_events", align_trake_events, METH_VARARGS, "Run ordered TRAKE DP for one video."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "C++17 acceleration kernels for Unified Video Retrieval.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&module);
}
