warning_flags = -Wall -Wextra -Wshadow
common_cflags = -O2 $(MFLAGS) $(STACKREALIGN) $(warning_flags)
AM_CXXFLAGS = -std=c++11 $(common_cflags)

AM_CPPFLAGS = $(VapourSynth_CFLAGS)

lib_LTLIBRARIES = libdegrainmedian.la

libdegrainmedian_la_SOURCES = src/degrainmedian.cpp

libdegrainmedian_la_LDFLAGS = -no-undefined -avoid-version $(PLUGINLDFLAGS)
