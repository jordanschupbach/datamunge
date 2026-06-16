

# Org → Markdown (with Babel)

This page is authored in Org mode and exported to Markdown via \`docs/org-to-md.el\`.
During export, all Babel code blocks are executed and their results are embedded
into the generated Markdown in \`docs/pages/\`.


# Examples (by language)

Each section below contains the canonical \`octra<sub>ex</sub>.\*\` example for that language.
During \`just prebuild-docs-pages\`, hidden shell blocks execute these examples
best-effort (printing \`SKIP\` when the relevant runtime/binding is unavailable).


## Sanity (executed)

    (+ 40 2)

    echo "Org Babel is executing blocks."


## C++

    #include <octra/octra.hpp>
    
    int main(void) {
      octra::hello();
      return 0;
    }

    Hello octra


## Python

    from pyoctra import octra
    
    # dir(octra)
    
    octra.hello()
    
    dp = octra.DPair(1, 2)
    
    s = octra.SVector(3)
    s[0] = "1"
    s[1] = "2"
    s[2] = "3"
    # s[3] = "4"  # <- Gets index out of bounds error
    
    
    class TimesTwo(octra.Callback):
        def call(self, x):
            return x * 2.0
    
    
    class TimesThree(octra.Callback):
        def call(self, x):
            return x * 3.0
    
    
    # cb = TimesTwo()
    cb = TimesThree()
    print("call_with_callback(3.0) =", octra.call_with_callback(3.0, cb))
    v = octra.make_dvector(1.0, 2.0, 3.0)
    v2 = octra.map_dvector_with_callback(v, cb)
    print("sum_dvector(map_dvector_with_callback(1,2,3)) =", octra.sum_dvector(v2))

    Hello octra
    call_with_callback(3.0) = 9.0
    sum_dvector(map_dvector_with_callback(1,2,3)) = 18.0


## JavaScript (Node)

    // Local dev: build first (`just build-javascript`), then run this example.
    const octra = require("../../index.js");
    
    octra.hello();
    
    // Note: SWIG's Node/JavaScript backend does not support directors/virtual-method
    // overrides the same way as Python/Ruby/Perl. `Callback` can still be passed,
    // but the default implementation is identity.
    const cb = new octra.Callback();
    console.log("call_with_callback(3.0) =", octra.call_with_callback(3.0, cb));
    const v2 = octra.map_dvector_with_callback(octra.make_dvector(1.0, 2.0, 3.0), cb);
    console.log("sum_dvector(map_dvector_with_callback) =", octra.sum_dvector(v2));
    
    // Bridging: pass a JS function into native code (via C callback trampoline).
    console.log("call_with_function(3.0) =", octra.call_with_function(3.0, (x) => x * 2.0));
    const out = octra.map_array_with_function([1.0, 2.0, 3.0], (x) => x * 2.0);
    console.log("map_array_with_function([1,2,3]) =", out);

    Hello octra
    call_with_callback(3.0) = 3
    sum_dvector(map_dvector_with_callback) = 6
    call_with_function(3.0) = 6
    map_array_with_function([1,2,3]) = [ 2, 4, 6 ]


## R

    library(octrar)
    
    octrar::hello()
    
    # Can only coerce (can't use methods directly without a wrapper)
    v <- octrar::DVector(c(0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0))
    
    # This works as expected
    p <- octrar::DPair(1.1, 2.2)
    p$first
    p$second
    
    # This works as expected
    p <- octrar::IPair(1.1, 2.2) # converts to integer
    p$first
    p$second

    Hello octra
    NULL
    [1] 1.1
    [1] 2.2
    [1] 1
    [1] 2


## Ruby

    require "octruby"
    
    Octra.hello
    
    dp = Octra::DPair.new(1.0, 2.0)
    dp.first
    dp.second
    puts "DPair created: #{dp.inspect}"
    dp.delete if dp.respond_to?(:delete)
    
    class TimesTwo < Octra::Callback
      def call(x)
        x * 2.0
      end
    end
    
    cb = TimesTwo.new
    puts "call_with_callback(3.0) = #{Octra.call_with_callback(3.0, cb)}"
    v = Octra.make_dvector(1.0, 2.0, 3.0)
    v2 = Octra.map_dvector_with_callback(v, cb)
    puts "sum_dvector(map_dvector_with_callback(1,2,3)) = #{Octra.sum_dvector(v2)}"

    Hello octra
    DPair created: std::pair<double,double > (1.0,2.0)
    call_with_callback(3.0) = 6.0
    sum_dvector(map_dvector_with_callback(1,2,3)) = 12.0


## Perl

    use strict;
    use warnings;
    
    use Octra;
    
    Octra::hello();
    
    my $dp = Octra::DPair->new(1.0, 2.0);
    print "DPair: First=" . $dp->swig_first_get() . ", Second=" . $dp->swig_second_get() . "\n";
    
    {
      package TimesTwo;
      our @ISA = ('Octra::Callback');
    
      sub call {
        my ($self, $x) = @_;
        return $x * 2.0;
      }
    }
    
    my $cb = TimesTwo->new();
    print "call_with_callback(3.0) = ", Octra::call_with_callback(3.0, $cb), "\n";
    my $v = Octra::make_dvector(1.0, 2.0, 3.0);
    my $v2 = Octra::map_dvector_with_callback($v, $cb);
    print "sum_dvector(map_dvector_with_callback(1,2,3)) = ", Octra::sum_dvector($v2), "\n";

    Hello octra
    DPair: First=1, Second=2
    call_with_callback(3.0) = 6
    sum_dvector(map_dvector_with_callback(1,2,3)) = 12


## PHP

    <?php
    
    
    $n = 100;
    $v = new DVector($n);
    for($i = 0; $i < 10; $i += 1) {
        $v->set($i, $i * 1.5);
    }
    for($i = 0; $i < 10; $i += 1) {
        print($v->get($i) . "\n");
    }
    
    $v2 = new IVector($n);
    for($i = 0; $i < 10; $i += 1) {
        $v2->set($i, $i * 1.5); // NOTE: Type coercion happens implicitly
    }
    
    for($i = 0; $i < 10; $i += 1) {
        print($v2->get($i) . "\n");
    }
    
    hello()
    
    ?>

    0
    1.5
    3
    4.5
    6
    7.5
    9
    10.5
    12
    13.5
    0
    1
    3
    4
    6
    7
    9
    10
    12
    13
    Hello octra


## Lua

    local octra = require("octra")
    octra.hello()
    
    local v = octra.make_dvector(1.0, 2.0, 3.0)
    print("sum_dvector:", octra.sum_dvector(v))
    
    local p = octra.make_dpair(1.25, 2.75)
    print("sum_dpair:", octra.sum_dpair(p))
    
    local cb = octra.Callback()
    print("call_with_callback(3.0):", octra.call_with_callback(3.0, cb))
    local v2 = octra.map_dvector_with_callback(octra.make_dvector(1.0, 2.0, 3.0), cb)
    print("sum_dvector(map_dvector_with_callback(1,2,3)):", octra.sum_dvector(v2))

    Hello octra
    sum_dvector:	6.0
    sum_dpair:	4.0
    call_with_callback(3.0):	3.0
    sum_dvector(map_dvector_with_callback(1,2,3)):	6.0


## Tcl

    package require Octra 0.0.1
    
    octra::hello
    
    set v [octra::new_DVector]
    octra::DVector_push_back $v 1.25
    octra::DVector_push_back $v 2.5
    puts "DVector size: [octra::DVector_size $v]"
    puts [format {DVector[0]: %s} [octra::DVector_get $v 0]]
    octra::delete_DVector $v
    
    set p [octra::new_DPair 3.0 4.5]
    puts "DPair: ([octra::DPair_first_get $p], [octra::DPair_second_get $p])"
    octra::delete_DPair $p
    
    puts "Octra Tcl example ran successfully."

    Octra Tcl example ran successfully.


## Octave

    1;
    
    octra;
    
    hello ();
    
    v = DVector ();
    v.append (1.0);
    v.append (2.0);
    v.append (3.5);
    printf ("vector size: %d\n", v.size ());
    printf ("vector[0]=%f vector[1]=%f\n", v.__paren__ (0), v.__paren__ (1));
    
    p = DPair (10.0, 25.5);
    printf ("pair sum: %f\n", p.first + p.second);

    Hello octra
    vector size: 3
    vector[0]=1.000000 vector[1]=2.000000
    pair sum: 35.500000


## Guile

    (use-modules (octra))
    
    (display "Hello from Guile + Octra\n")
    (hello)
    
    (let ((v (make_dvector 10.0 20.0 30.0)))
      (format #t "sum_dvector = ~a\n" (sum_dvector v)))
    
    (let ((p (make_dpair 4.0 5.0)))
      (format #t "sum_dpair = ~a\n" (sum_dpair p)))

    Hello octra
    Hello from Guile + Octra
    sum_dvector = 60.0
    sum_dpair = 9.0


## OCaml

    let () =
      let open Swig in
      ignore (Octra._hello C_void);
      print_endline "octra: _hello() ok";
    
      let p = Octra.new_DPair (C_list [ C_double 1.25; C_double 2.5 ]) in
      let p_first = get_float (invoke p "[first]" C_void) in
      let p_second = get_float (invoke p "[second]" C_void) in
      let p_sum = get_float (Octra._sum_dpair p) in
      Printf.printf "DPair: first=%g second=%g sum=%g\n" p_first p_second p_sum;
    
      let v = Octra.new_DVector C_void in
      ignore (invoke v "push_back" (C_double 3.0));
      ignore (invoke v "push_back" (C_double 4.5));
      let v0 = get_float (invoke v "[]" (C_int 0)) in
      let v1 = get_float (invoke v "[]" (C_int 1)) in
      let v_sum = get_float (Octra._sum_dvector v) in
      Printf.printf "DVector: [0]=%g [1]=%g sum=%g\n" v0 v1 v_sum

    Hello octra
    octra: _hello() ok
    DPair: first=1.25 second=2.5 sum=3.75
    DVector: [0]=3 [1]=4.5 sum=7.5


## Go

    package main
    
    import (
            "fmt"
            "octra"
    )
    
    func main() {
            octra.Hello()
    
            dp := octra.NewDPair(1.0, 2.0)
            fmt.Printf("DPair: First=%v, Second=%v\n", dp.GetFirst(), dp.GetSecond())
    
            s := octra.NewSVector(int64(3))
            s.Set(0, "1")
            s.Set(1, "2")
            s.Set(2, "3")
            fmt.Printf("SVector: [0]=%v, [1]=%v, [2]=%v\n", s.Get(0), s.Get(1), s.Get(2))
    
            // Clean up
            octra.DeleteDPair(dp)
            octra.DeleteSVector(s)
    }

    Hello octra
    DPair: First=1, Second=2
    SVector: [0]=1, [1]=2, [2]=3


# Using Octra

See \`README.md\` for language bindings and build/test commands.

