#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include <VapourSynth.h>
#include <VSHelper.h>


typedef void (*DegrainFunction) (const uint8_t *prevp, const uint8_t *srcp, const uint8_t *nextp, uint8_t *dstp, int stride, int width, int height, int limit);


static inline void checkBetterNeighbours(int a, int b, int &diff, int &min, int &max) {
    int new_diff = std::abs(a - b);

    if (new_diff <= diff) {
        diff = new_diff;
        min = std::min(a, b);
        max = std::max(a, b);
    }
}


static void mode0(const uint8_t *prevp, const uint8_t *srcp, const uint8_t *nextp, uint8_t *dstp, int stride, int width, int height, int limit, int interlaced) {
    const int distance = stride << interlaced;
    const int skip_rows = 1 << interlaced;

    // Copy first line(s).
    memcpy(dstp, srcp, width);
    if (interlaced)
        memcpy(dstp + stride, srcp + stride, width);
    prevp += distance;
    srcp += distance;
    nextp += distance;
    dstp += distance;
    
    for (int y = skip_rows; y < height - skip_rows; y++) {
        dstp[0] = srcp[0];

        for (int x = 1; x < width - 1; x++) {
            int p1, p2, p3,
                p4, p5, p6,
                p7, p8, p9;
            
            int s1, s2, s3,
                s4, s5, s6,
                s7, s8, s9;
            
            int n1, n2, n3,
                n4, n5, n6,
                n7, n8, n9;
            
            p1 = prevp[x - distance - 1];
            p2 = prevp[x - distance];
            p3 = prevp[x - distance + 1];
            p4 = prevp[x - 1];
            p5 = prevp[x];
            p6 = prevp[x + 1];
            p7 = prevp[x + distance - 1];
            p8 = prevp[x + distance];
            p9 = prevp[x + distance + 1];
            
            s1 = srcp[x - distance - 1];
            s2 = srcp[x - distance];
            s3 = srcp[x - distance + 1];
            s4 = srcp[x - 1];
            s5 = srcp[x];
            s6 = srcp[x + 1];
            s7 = srcp[x + distance - 1];
            s8 = srcp[x + distance];
            s9 = srcp[x + distance + 1];
            
            n1 = nextp[x - distance - 1];
            n2 = nextp[x - distance];
            n3 = nextp[x - distance + 1];
            n4 = nextp[x - 1];
            n5 = nextp[x];
            n6 = nextp[x + 1];
            n7 = nextp[x + distance - 1];
            n8 = nextp[x + distance];
            n9 = nextp[x + distance + 1];
            
            int diff = 255;
            int min = 0;
            int max = 255;

            checkBetterNeighbours(n1, p9, diff, min, max);
            checkBetterNeighbours(n3, p7, diff, min, max);
            checkBetterNeighbours(n7, p3, diff, min, max);
            checkBetterNeighbours(n9, p1, diff, min, max);
            checkBetterNeighbours(n8, p2, diff, min, max);
            checkBetterNeighbours(n2, p8, diff, min, max);
            checkBetterNeighbours(n4, p6, diff, min, max);
            checkBetterNeighbours(n6, p4, diff, min, max);
            checkBetterNeighbours(n5, p5, diff, min, max);

            checkBetterNeighbours(s1, s9, diff, min, max);
            checkBetterNeighbours(s3, s7, diff, min, max);
            checkBetterNeighbours(s2, s8, diff, min, max);
            checkBetterNeighbours(s4, s6, diff, min, max);

            int result = std::max(min, std::min(s5, max));

            dstp[x] = std::max(s5 - limit, std::min(result, s5 + limit));
        }

        dstp[width - 1] = srcp[width - 1];
        
        prevp += stride;
        srcp += stride;
        nextp += stride;
        dstp += stride;
    }
    
    // Copy last line(s).
    memcpy(dstp, srcp, width);
    if (interlaced)
        memcpy(dstp + stride, srcp + stride, width);
}


typedef struct DegrainMedianData {
    VSNodeRef *clip;
    const VSVideoInfo *vi;

    int limit[3];
    int mode[3];
    bool interlaced;
    bool norow;

    DegrainFunction degrain[3];
} DegrainMedianData;


static void VS_CC degrainMedianInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    (void)in;
    (void)out;
    (void)core;

    DegrainMedianData *d = (DegrainMedianData *) * instanceData;

    vsapi->setVideoInfo(d->vi, 1, node);
}


static void selectFunctions(const VSFormat *fmt, const int *limit, const int *mode, bool interlaced, bool norow, DegrainFunction *degrain) {
    for (int plane = 0; plane < fmt->numPlanes; plane++) {
        if (limit[plane] == 0) {
            degrain[plane] = nullptr;
            continue;
        }

        /// select
    }
}


static const VSFrameRef *VS_CC degrainMedianGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    (void)frameData;

    const DegrainMedianData *d = (const DegrainMedianData *) * instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(std::max(0, n - 1), d->clip, frameCtx);
        vsapi->requestFrameFilter(n, d->clip, frameCtx);
        vsapi->requestFrameFilter(std::min(n + 1, d->vi->numFrames - 1), d->clip, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        if (n == 0 || n == d->vi->numFrames - 1)
            return vsapi->getFrameFilter(n, d->clip, frameCtx);

        const VSFrameRef *prev = vsapi->getFrameFilter(n - 1, d->clip, frameCtx);
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->clip, frameCtx);
        const VSFrameRef *next = vsapi->getFrameFilter(n + 1, d->clip, frameCtx);

        const VSFormat *prev_fmt = vsapi->getFrameFormat(prev);
        const VSFormat *src_fmt = vsapi->getFrameFormat(src);
        const VSFormat *next_fmt = vsapi->getFrameFormat(next);

        if (prev_fmt != src_fmt || next_fmt != src_fmt ||
            vsapi->getFrameWidth(prev, 0) != vsapi->getFrameWidth(src, 0) ||
            vsapi->getFrameWidth(next, 0) != vsapi->getFrameWidth(src, 0) ||
            vsapi->getFrameHeight(prev, 0) != vsapi->getFrameHeight(src, 0) ||
            vsapi->getFrameHeight(next, 0) != vsapi->getFrameHeight(src, 0)) {
            vsapi->freeFrame(prev);
            vsapi->freeFrame(next);
            return src;
        }


        DegrainFunction degrain[3] = { d->degrain[0], d->degrain[1], d->degrain[2] };
        if (!d->vi->format)
            selectFunctions(src_fmt, d->limit, d->mode, d->interlaced, d->norow, degrain);


        const VSFrameRef *frames[3] = {
            d->limit[0] == 0 ? src : nullptr,
            d->limit[1] == 0 ? src : nullptr,
            d->limit[2] == 0 ? src : nullptr
        };
        int planes[3] = { 0, 1, 2 };

        VSFrameRef *dst = vsapi->newVideoFrame2(src_fmt, vsapi->getFrameWidth(src, 0), vsapi->getFrameHeight(src, 0), frames, planes, src, core);


        int pixel_max = (1 << src_fmt->bitsPerSample) - 1;

        for (int plane = 0; plane < src_fmt->numPlanes; plane++) {
            if (d->limit[plane] == 0)
                continue;

            const uint8_t *prevp = vsapi->getReadPtr(prev, plane);
            const uint8_t *srcp = vsapi->getReadPtr(src, plane);
            const uint8_t *nextp = vsapi->getReadPtr(next, plane);
            uint8_t *dstp = vsapi->getWritePtr(dst, plane);

            int stride = vsapi->getStride(src, plane);
            int width = vsapi->getFrameWidth(src, plane);
            int height = vsapi->getFrameHeight(src, plane);

            int limit = pixel_max * d->limit[plane] / 255;

            degrain[plane](prevp, srcp, nextp, dstp, stride, width, height, limit);
        }


        vsapi->freeFrame(prev);
        vsapi->freeFrame(src);
        vsapi->freeFrame(next);

        return dst;
    }

    return NULL;
}


static void VS_CC degrainMedianFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    (void)core;

    DegrainMedianData *d = (DegrainMedianData *)instanceData;

    vsapi->freeNode(d->clip);
    free(d);
}


static void VS_CC degrainMedianCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    (void)userData;

    DegrainMedianData d;
    DegrainMedianData *data;

    int err;

    d.interlaced = !!vsapi->propGetInt(in, "interlaced", 0, &err);

    d.norow = !!vsapi->propGetInt(in, "norow", 0, &err);

    d.clip = vsapi->propGetNode(in, "clip", 0, NULL);
    d.vi = vsapi->getVideoInfo(d.clip);

    if (d.vi->format && (d.vi->format->sampleType != stInteger || d.vi->format->bitsPerSample > 16)) {
        vsapi->setError(out, "DegrainMedian: only 8..16 bit clips are supported.");
        vsapi->freeNode(d.clip);
        return;
    }


    int num_limits = vsapi->propNumElements(in, "limit");
    if (d.vi->format && num_limits > d.vi->format->numPlanes) {
        vsapi->setError(out, "DegrainMedian: limit has more elements than there are planes.");
        vsapi->freeNode(d.clip);
        return;
    }

    if (num_limits == -1) {
        d.limit[0] = d.limit[1] = d.limit[2] = 4;
    } else {
        for (int i = 0; i < 3; i++) {
            d.limit[i] = int64ToIntS(vsapi->propGetInt(in, "limit", i, &err));
            if (err) {
                d.limit[i] = d.limit[i - 1];
            } else {
                if (d.limit[i] < 0 || d.limit[i] > 255) {
                    vsapi->setError(out, ("DegrainMedian: limit[" + std::to_string(i) + "] must be between 0 and 255 (inclusive).").c_str());
                    vsapi->freeNode(d.clip);
                    return;
                }
            }
        }
    }


    int num_modes = vsapi->propNumElements(in, "mode");
    if (d.vi->format && num_modes > d.vi->format->numPlanes) {
        vsapi->setError(out, "DegrainMedian: mode has more elements than there are planes.");
        vsapi->freeNode(d.clip);
        return;
    }

    if (num_modes == -1) {
        d.mode[0] = d.mode[1] = d.mode[2] = 1;
    } else {
        for (int i = 0; i < 3; i++) {
            d.mode[i] = int64ToIntS(vsapi->propGetInt(in, "mode", i, &err));
            if (err) {
                d.mode[i] = d.mode[i - 1];
            } else {
                if (d.mode[i] < 0 || d.mode[i] > 5) {
                    vsapi->setError(out, ("DegrainMedian: mode[" + std::to_string(i) + "] must be between 0 and 5 (inclusive).").c_str());
                    vsapi->freeNode(d.clip);
                    return;
                }
            }
        }
    }


    if (d.vi->format)
        selectFunctions(d.vi->format, d.limit, d.mode, d.interlaced, d.norow, d.degrain);


    data = (DegrainMedianData *)malloc(sizeof(d));
    *data = d;

    vsapi->createFilter(in, out, "DegrainMedian", degrainMedianInit, degrainMedianGetFrame, degrainMedianFree, fmParallel, 0, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.nodame.degrainmedian", "dgm", "Spatio-temporal limited median denoiser", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("DegrainMedian",
                 "clip:clip;"
                 "limit:int[]:opt;"
                 "mode:int[]:opt;"
                 "interlaced:int:opt;"
                 "norow:int:opt;"
                 , degrainMedianCreate, 0, plugin);
}
