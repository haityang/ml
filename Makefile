CXX ?= g++

CXXFLAGS += -c -Wall $(shell pkg-config --cflags opencv4) 
LDFLAGS += $(shell pkg-config --libs --static opencv4)

all:example 

example: demo_bayes.o; $(CXX) $< -o $@ $(LDFLAGS)

%.o: %.cpp; $(CXX) $< -o $@ $(CXXFLAGS)

clean: ; rm -f demo_bayes.o example
