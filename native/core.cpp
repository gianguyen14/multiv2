#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct State {
    double score = -std::numeric_limits<double>::infinity();
    std::vector<long long> path;
    bool valid = false;
};

bool path_less(const std::vector<long long>& a, const std::vector<long long>& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

bool better_state(const State& candidate, const State& current) {
    if (!candidate.valid) {
        return false;
    }
    if (!current.valid) {
        return true;
    }
    if (candidate.score > current.score) {
        return true;
    }
    if (candidate.score < current.score) {
        return false;
    }
    // Python reference implementation uses max((score, [-frame, ...])).
    // For equal scores that means the lexicographically smaller frame path wins.
    return path_less(candidate.path, current.path);
}

bool sequence_to_int64(PyObject* obj, std::vector<long long>& out, const char* name) {
    PyObject* seq = PySequence_Fast(obj, name);
    if (!seq) {
        return false;
    }
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    out.reserve(static_cast<std::size_t>(n));
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        long long value = PyLong_AsLongLong(items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return false;
        }
        out.push_back(value);
    }
    Py_DECREF(seq);
    return true;
}

bool sequence_to_double(PyObject* obj, std::vector<double>& out, const char* name) {
    PyObject* seq = PySequence_Fast(obj, name);
    if (!seq) {
        return false;
    }
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    out.reserve(static_cast<std::size_t>(n));
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        double value = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return false;
        }
        out.push_back(value);
    }
    Py_DECREF(seq);
    return true;
}

PyObject* py_version(PyObject*, PyObject*) {
    return PyUnicode_FromString("0.1.0");
}

PyObject* py_smooth_scores(PyObject*, PyObject* args) {
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

    const bool format_f32 = view.itemsize == 4 && view.format && std::strchr(view.format, 'f');
    const bool format_f64 = view.itemsize == 8 && view.format && std::strchr(view.format, 'd');
    if (!format_f32 && !format_f64) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "raw_scores must be float32 or float64");
        return nullptr;
    }

    const Py_ssize_t n = view.len / view.itemsize;
    PyObject* output = PyBytes_FromStringAndSize(nullptr, n * static_cast<Py_ssize_t>(sizeof(float)));
    if (!output) {
        PyBuffer_Release(&view);
        return nullptr;
    }
    float* out = reinterpret_cast<float*>(PyBytes_AS_STRING(output));

    auto value_at = [&](Py_ssize_t idx) -> double {
        if (format_f32) {
            const float* values = static_cast<const float*>(view.buf);
            return static_cast<double>(values[idx]);
        }
        const double* values = static_cast<const double*>(view.buf);
        return values[idx];
    };

    for (Py_ssize_t j = 0; j < n; ++j) {
        const Py_ssize_t begin = std::max<Py_ssize_t>(0, j - pool_window);
        const Py_ssize_t end = std::min<Py_ssize_t>(n, j + pool_window + 1);
        double sum = 0.0;
        for (Py_ssize_t k = begin; k < end; ++k) {
            sum += value_at(k);
        }
        const double local_mean = sum / static_cast<double>(end - begin);
        const double score = weight_visual * value_at(j) + weight_temporal * local_mean;
        out[j] = static_cast<float>(score);
    }

    PyBuffer_Release(&view);
    return output;
}

PyObject* py_temporal_nms_indices(PyObject*, PyObject* args) {
    PyObject* video_ids_obj = nullptr;
    PyObject* frame_ids_obj = nullptr;
    long long min_gap = 0;
    Py_ssize_t top_k = 0;
    if (!PyArg_ParseTuple(args, "OOLL", &video_ids_obj, &frame_ids_obj, &min_gap, &top_k)) {
        return nullptr;
    }
    if (min_gap < 0 || top_k < 0) {
        PyErr_SetString(PyExc_ValueError, "min_gap and top_k must be >= 0");
        return nullptr;
    }

    PyObject* videos = PySequence_Fast(video_ids_obj, "video_ids must be a sequence");
    if (!videos) {
        return nullptr;
    }
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

    PyObject* selected_indices = PyList_New(0);
    if (!selected_indices) {
        Py_DECREF(videos);
        Py_DECREF(frames);
        return nullptr;
    }

    std::unordered_map<std::string, std::vector<long long>> selected_by_video;
    PyObject** video_items = PySequence_Fast_ITEMS(videos);
    PyObject** frame_items = PySequence_Fast_ITEMS(frames);

    for (Py_ssize_t i = 0; i < n && PyList_GET_SIZE(selected_indices) < top_k; ++i) {
        const char* video_id = PyUnicode_AsUTF8(video_items[i]);
        if (!video_id) {
            Py_DECREF(selected_indices);
            Py_DECREF(videos);
            Py_DECREF(frames);
            return nullptr;
        }
        const long long frame_id = PyLong_AsLongLong(frame_items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(selected_indices);
            Py_DECREF(videos);
            Py_DECREF(frames);
            return nullptr;
        }

        auto& chosen = selected_by_video[video_id];
        bool too_close = false;
        for (long long prior : chosen) {
            if (std::llabs(frame_id - prior) < min_gap) {
                too_close = true;
                break;
            }
        }
        if (!too_close) {
            chosen.push_back(frame_id);
            PyObject* index_obj = PyLong_FromSsize_t(i);
            if (!index_obj || PyList_Append(selected_indices, index_obj) != 0) {
                Py_XDECREF(index_obj);
                Py_DECREF(selected_indices);
                Py_DECREF(videos);
                Py_DECREF(frames);
                return nullptr;
            }
            Py_DECREF(index_obj);
        }
    }

    Py_DECREF(videos);
    Py_DECREF(frames);
    return selected_indices;
}

struct MergedRegion {
    long long start = 0;
    long long end = 0;
    std::vector<long long> candidate_frames;
    double max_score = 0.0;
};

PyObject* py_merge_temporal_regions(PyObject*, PyObject* args) {
    PyObject* frame_ids_obj = nullptr;
    PyObject* scores_obj = nullptr;
    long long delta_frames = 0;
    long long total_frames = 0;
    Py_ssize_t max_regions = 0;
    if (!PyArg_ParseTuple(args, "OOLLL", &frame_ids_obj, &scores_obj, &delta_frames, &total_frames, &max_regions)) {
        return nullptr;
    }
    if (delta_frames < 0 || total_frames <= 0 || max_regions < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid temporal-region limits");
        return nullptr;
    }

    std::vector<long long> frame_ids;
    std::vector<double> scores;
    if (!sequence_to_int64(frame_ids_obj, frame_ids, "frame_ids must be a sequence") ||
        !sequence_to_double(scores_obj, scores, "scores must be a sequence")) {
        return nullptr;
    }
    if (frame_ids.size() != scores.size()) {
        PyErr_SetString(PyExc_ValueError, "frame_ids and scores must have the same length");
        return nullptr;
    }

    struct Interval {
        long long start;
        long long end;
        long long frame_id;
        double score;
    };

    std::vector<Interval> intervals;
    intervals.reserve(frame_ids.size());
    for (std::size_t i = 0; i < frame_ids.size(); ++i) {
        const long long frame_id = frame_ids[i];
        const long long start = std::max<long long>(0, frame_id - delta_frames);
        const long long end = std::min<long long>(total_frames - 1, frame_id + delta_frames);
        intervals.push_back({start, end, frame_id, scores[i]});
    }
    std::stable_sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
        return a.start < b.start;
    });

    std::vector<MergedRegion> merged;
    for (const auto& interval : intervals) {
        if (merged.empty() || interval.start > merged.back().end) {
            MergedRegion region;
            region.start = interval.start;
            region.end = interval.end;
            region.candidate_frames.push_back(interval.frame_id);
            region.max_score = interval.score;
            merged.push_back(std::move(region));
        } else {
            auto& region = merged.back();
            region.end = std::max(region.end, interval.end);
            region.candidate_frames.push_back(interval.frame_id);
            region.max_score = std::max(region.max_score, interval.score);
        }
    }

    if (merged.size() > static_cast<std::size_t>(max_regions)) {
        std::stable_sort(merged.begin(), merged.end(), [](const MergedRegion& a, const MergedRegion& b) {
            return a.max_score > b.max_score;
        });
        merged.resize(static_cast<std::size_t>(max_regions));
        std::stable_sort(merged.begin(), merged.end(), [](const MergedRegion& a, const MergedRegion& b) {
            return a.start < b.start;
        });
    }

    PyObject* output = PyList_New(static_cast<Py_ssize_t>(merged.size()));
    if (!output) {
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(merged.size()); ++i) {
        const auto& region = merged[static_cast<std::size_t>(i)];
        PyObject* frames_list = PyList_New(static_cast<Py_ssize_t>(region.candidate_frames.size()));
        if (!frames_list) {
            Py_DECREF(output);
            return nullptr;
        }
        for (Py_ssize_t j = 0; j < static_cast<Py_ssize_t>(region.candidate_frames.size()); ++j) {
            PyList_SET_ITEM(frames_list, j, PyLong_FromLongLong(region.candidate_frames[static_cast<std::size_t>(j)]));
        }

        PyObject* item = PyTuple_New(4);
        if (!item) {
            Py_DECREF(frames_list);
            Py_DECREF(output);
            return nullptr;
        }
        PyTuple_SET_ITEM(item, 0, PyLong_FromLongLong(region.start));
        PyTuple_SET_ITEM(item, 1, PyLong_FromLongLong(region.end));
        PyTuple_SET_ITEM(item, 2, frames_list);
        PyTuple_SET_ITEM(item, 3, PyFloat_FromDouble(region.max_score));
        PyList_SET_ITEM(output, i, item);
    }
    return output;
}

PyObject* py_align_trake_events(PyObject*, PyObject* args) {
    PyObject* frames_events_obj = nullptr;
    PyObject* scores_events_obj = nullptr;
    double transition_penalty = 0.0;
    PyObject* max_gap_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOdO", &frames_events_obj, &scores_events_obj, &transition_penalty, &max_gap_obj)) {
        return nullptr;
    }

    bool has_max_gap = max_gap_obj != Py_None;
    long long max_gap = 0;
    if (has_max_gap) {
        max_gap = PyLong_AsLongLong(max_gap_obj);
        if (PyErr_Occurred()) {
            return nullptr;
        }
        if (max_gap < 0) {
            PyErr_SetString(PyExc_ValueError, "max_gap must be >= 0 or None");
            return nullptr;
        }
    }

    PyObject* frames_events = PySequence_Fast(frames_events_obj, "frame_ids_by_event must be a sequence");
    if (!frames_events) {
        return nullptr;
    }
    PyObject* scores_events = PySequence_Fast(scores_events_obj, "scores_by_event must be a sequence");
    if (!scores_events) {
        Py_DECREF(frames_events);
        return nullptr;
    }

    const Py_ssize_t event_count = PySequence_Fast_GET_SIZE(frames_events);
    if (event_count == 0 || PySequence_Fast_GET_SIZE(scores_events) != event_count) {
        Py_DECREF(frames_events);
        Py_DECREF(scores_events);
        if (event_count == 0) {
            Py_RETURN_NONE;
        }
        PyErr_SetString(PyExc_ValueError, "event frame/score lists must have matching lengths");
        return nullptr;
    }

    struct Candidate {
        long long frame;
        double score;
        std::size_t order;
    };
    std::vector<std::vector<Candidate>> events;
    events.reserve(static_cast<std::size_t>(event_count));

    PyObject** frame_event_items = PySequence_Fast_ITEMS(frames_events);
    PyObject** score_event_items = PySequence_Fast_ITEMS(scores_events);
    for (Py_ssize_t e = 0; e < event_count; ++e) {
        std::vector<long long> frames;
        std::vector<double> scores;
        if (!sequence_to_int64(frame_event_items[e], frames, "event frame IDs must be a sequence") ||
            !sequence_to_double(score_event_items[e], scores, "event scores must be a sequence")) {
            Py_DECREF(frames_events);
            Py_DECREF(scores_events);
            return nullptr;
        }
        if (frames.empty()) {
            Py_DECREF(frames_events);
            Py_DECREF(scores_events);
            Py_RETURN_NONE;
        }
        if (frames.size() != scores.size()) {
            Py_DECREF(frames_events);
            Py_DECREF(scores_events);
            PyErr_SetString(PyExc_ValueError, "each event must have matching frame and score lengths");
            return nullptr;
        }
        std::vector<Candidate> candidates;
        candidates.reserve(frames.size());
        for (std::size_t i = 0; i < frames.size(); ++i) {
            candidates.push_back({frames[i], scores[i], i});
        }
        std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.frame < b.frame;
        });
        events.push_back(std::move(candidates));
    }

    Py_DECREF(frames_events);
    Py_DECREF(scores_events);

    std::vector<State> states;
    states.reserve(events[0].size());
    for (const auto& candidate : events[0]) {
        State state;
        state.score = candidate.score;
        state.path = {candidate.frame};
        state.valid = true;
        states.push_back(std::move(state));
    }

    for (std::size_t e = 1; e < events.size(); ++e) {
        std::vector<State> next_states(events[e].size());
        for (std::size_t j = 0; j < events[e].size(); ++j) {
            const auto& current_candidate = events[e][j];
            State best;
            for (std::size_t i = 0; i < events[e - 1].size(); ++i) {
                if (!states[i].valid) {
                    continue;
                }
                const long long gap = current_candidate.frame - events[e - 1][i].frame;
                if (gap <= 0 || (has_max_gap && gap > max_gap)) {
                    continue;
                }
                State proposal;
                proposal.valid = true;
                proposal.score = states[i].score + current_candidate.score - transition_penalty * static_cast<double>(gap);
                proposal.path = states[i].path;
                proposal.path.push_back(current_candidate.frame);
                if (better_state(proposal, best)) {
                    best = std::move(proposal);
                }
            }
            next_states[j] = std::move(best);
        }
        states = std::move(next_states);
    }

    State best;
    for (const auto& state : states) {
        if (state.valid && state.path.size() == events.size() && better_state(state, best)) {
            best = state;
        }
    }
    if (!best.valid) {
        Py_RETURN_NONE;
    }

    PyObject* path = PyList_New(static_cast<Py_ssize_t>(best.path.size()));
    if (!path) {
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(best.path.size()); ++i) {
        PyList_SET_ITEM(path, i, PyLong_FromLongLong(best.path[static_cast<std::size_t>(i)]));
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

PyMethodDef module_methods[] = {
    {"version", py_version, METH_NOARGS, "Return native-core ABI version."},
    {"smooth_scores", py_smooth_scores, METH_VARARGS, "Smooth temporal scores and return float32 bytes."},
    {"temporal_nms_indices", py_temporal_nms_indices, METH_VARARGS, "Return selected indices for temporal NMS."},
    {"merge_temporal_regions", py_merge_temporal_regions, METH_VARARGS, "Merge overlapping temporal candidate windows."},
    {"align_trake_events", py_align_trake_events, METH_VARARGS, "Run ordered TRAKE dynamic programming for one video."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "C++17 acceleration kernels for Unified Video Retrieval.",
    -1,
    module_methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&module_def);
}
