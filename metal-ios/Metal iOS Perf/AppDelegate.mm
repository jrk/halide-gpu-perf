//
//  AppDelegate.m
//  Metal iOS Perf
//
//  Created by Jonathan Ragan-Kelley on 1/10/17.
//  Copyright © 2017 Halide. All rights reserved.
//

#import "AppDelegate.h"
#import "ViewController.h"

#include "HalideRuntime.h"
#include "HalideRuntimeMetal.h"

#include "reaction_diffusion_init_metal.h"
#include "reaction_diffusion_render_metal.h"
#include "reaction_diffusion_update_metal.h"
#include "reaction_diffusion_init_arm.h"
#include "reaction_diffusion_render_arm.h"
#include "reaction_diffusion_update_arm.h"

#include <algorithm>

static const int test_width = 1024;
static const int test_height = 1024;
static const int test_iterations = 100;

@interface AppDelegate ()

@end

@implementation AppDelegate
{
@private
    // TODO: put all this inside a struct def, with one for each target tested
    struct buffer_t buf1, buf1_cpu;
    struct buffer_t buf2, buf2_cpu;
    struct buffer_t pixel_buf, pixel_buf_cpu;
    
    int32_t iteration;
    
    float cx, cy;
    
    double lastFrameTime;
    double frameElapsedEstimate;
    double msPerIter;
}

- (void)runBench:(BOOL)metal {
    
    float startTime = CACurrentMediaTime();
    for (int i = 0; i < test_iterations; i++) {
        float tx = -100, ty = -100; // arbitrary
        if (metal) {
            reaction_diffusion_update_metal((__bridge void *)self, &buf1, tx, ty, cx, cy, iteration++, &buf2);
            reaction_diffusion_render_metal((__bridge void *)self, &buf2, &pixel_buf);
        } else {
            reaction_diffusion_update_arm((__bridge void *)self, &buf1_cpu, tx, ty, cx, cy, iteration++, &buf2_cpu);
            reaction_diffusion_render_arm((__bridge void *)self, &buf2_cpu, &pixel_buf_cpu);
        }
        
        std::swap(buf1, buf2);
    }
    
    auto complete = ^(void) {
        lastFrameTime = CACurrentMediaTime() - startTime;
        msPerIter = 1000*lastFrameTime/test_iterations;
        [self logPerf:metal];
        [self dispatchNextBench:metal]; // repeat...
    };

    // HACK: CPU tasks also enqueue on the Metal command queue, for serialization of tests
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    
    [commandBuffer addCompletedHandler: ^(id<MTLCommandBuffer>) { complete(); }];
    [commandBuffer enqueue];
    [commandBuffer commit];
    [_commandQueue insertDebugCaptureBoundary];
}

- (void)logPerf:(BOOL)metal {
    NSLog(@"%s: Completed after %f\n"
          "%f ms/iteration",
          metal ? "METAL" : "CPU", lastFrameTime, msPerIter);

    // UI updates have to happen on the main thread
    dispatch_async(dispatch_get_main_queue(), ^(void) {
        ViewController *vc = (ViewController*)self.window.rootViewController;
        [vc setRuntime:msPerIter forMetal:metal];
    });
}

- (void)dispatchNextBench:(BOOL)metal {
    
    //dispatch_async(
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 1000000000),
                   dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                    ^(void) {
                        [self runBench:!metal];
                    });
}

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    
    _device = MTLCreateSystemDefaultDevice();
    _commandQueue = [_device newCommandQueue];
    
    {
        buf1 = {0};
        buf1.extent[0] = test_width;
        buf1.extent[1] = test_height;
        buf1.extent[2] = 3;
        buf1.stride[0] = 3;
        buf1.stride[1] = buf1.extent[0] * buf1.stride[0];
        buf1.stride[2] = 1;
        buf1.elem_size = sizeof(float);
        
        cx = test_width / 2.0;
        cy = test_height / 2.0;
        
        buf2 = buf1;
        buf1.host = (uint8_t *)malloc(4 * 3 * buf1.extent[0] * buf1.extent[1]);
        buf2.host = (uint8_t *)malloc(4 * 3 * buf2.extent[0] * buf2.extent[1]);
        // Destination buf must have rows a multiple of 64 bytes for Metal's copyFromBuffer method.
        pixel_buf = {0};
        pixel_buf.extent[0] = buf1.extent[0];
        pixel_buf.extent[1] = buf1.extent[1];
        pixel_buf.stride[0] = 1;
        pixel_buf.stride[1] = (pixel_buf.extent[1] + 63) & ~63;
        pixel_buf.elem_size = sizeof(uint32_t);
        pixel_buf.host = (uint8_t *)malloc(4 * pixel_buf.stride[1] * pixel_buf.extent[1]);
        
        NSLog(@"Calling reaction_diffusion_init_metal size (%u x %u)", buf1.extent[0], buf1.extent[1]);
        reaction_diffusion_init_metal((__bridge void *)self, cx, cy, &buf1);
        NSLog(@"Returned from reaction_diffusion_init_metal");
    }

    {
        int image_width = test_width,
            image_height = test_height;
        buf1_cpu = {0};
        buf1_cpu.extent[0] = image_width;
        buf1_cpu.extent[1] = image_height;
        buf1_cpu.extent[2] = 3;
        buf1_cpu.stride[0] = 1;
        buf1_cpu.stride[1] = image_width;
        buf1_cpu.stride[2] = image_width * image_height;
        buf1_cpu.elem_size = 4;
        
        uint32_t *pixels = (uint32_t *)malloc(4*image_width*image_height);

        buf2_cpu = buf1_cpu, pixel_buf_cpu = buf1_cpu;
        buf1_cpu.host = (uint8_t *)malloc(4 * 3 * image_width * image_height);
        buf2_cpu.host = (uint8_t *)malloc(4 * 3 * image_width * image_height);
        pixel_buf_cpu.extent[2] = pixel_buf_cpu.stride[2] = 0;
        pixel_buf_cpu.host = (uint8_t *)pixels;

        NSLog(@"Calling reaction_diffusion_init_arm size (%u x %u)", buf1_cpu.extent[0], buf1_cpu.extent[1]);
        reaction_diffusion_init_arm((__bridge void *)self, cx, cy, &buf1_cpu);
        NSLog(@"Returned from reaction_diffusion_init_arm");
    }
    
    iteration = 0;
    lastFrameTime = -1;
    frameElapsedEstimate = -1;
    
    dispatch_async(dispatch_get_main_queue(), ^(void) {
        ViewController *vc = (ViewController*)self.window.rootViewController;
        [vc setApp:@"Reaction Diffusion"];
        [vc setRuntime:-1.0f forMetal:true];
        [vc setRuntime:-1.0f forMetal:false];
    });
    
    // Start the bench loop on the global "interactive" GCD queue
    [self dispatchNextBench:false];
    
    return YES;
}


- (void)applicationWillResignActive:(UIApplication *)application {
    // Sent when the application is about to move from active to inactive state. This can occur for certain types of temporary interruptions (such as an incoming phone call or SMS message) or when the user quits the application and it begins the transition to the background state.
    // Use this method to pause ongoing tasks, disable timers, and invalidate graphics rendering callbacks. Games should use this method to pause the game.
}


- (void)applicationDidEnterBackground:(UIApplication *)application {
    // Use this method to release shared resources, save user data, invalidate timers, and store enough application state information to restore your application to its current state in case it is terminated later.
    // If your application supports background execution, this method is called instead of applicationWillTerminate: when the user quits.
}


- (void)applicationWillEnterForeground:(UIApplication *)application {
    // Called as part of the transition from the background to the active state; here you can undo many of the changes made on entering the background.
}


- (void)applicationDidBecomeActive:(UIApplication *)application {
    // Restart any tasks that were paused (or not yet started) while the application was inactive. If the application was previously in the background, optionally refresh the user interface.
}


- (void)applicationWillTerminate:(UIApplication *)application {
    // Called when the application is about to terminate. Save data if appropriate. See also applicationDidEnterBackground:.
}

@end

extern "C" {
    
    int halide_metal_acquire_context(void *user_context, halide_metal_device **device_ret,
                                     halide_metal_command_queue **queue_ret, bool create) {
        AppDelegate *app = (__bridge AppDelegate *)user_context;
        *device_ret = (__bridge halide_metal_device *)app.device;
        *queue_ret = (__bridge halide_metal_command_queue *)app.commandQueue;
        return 0;
    }
    
    int halide_metal_release_context(void *user_context) {
        return 0;
    }
}
