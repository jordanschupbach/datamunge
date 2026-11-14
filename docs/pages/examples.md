

# Org → Markdown (with Babel)

This page is authored in Org mode and exported to Markdown via \`docs/org-to-md.el\`.
During export, all Babel code blocks are executed and their results are embedded
into the generated Markdown in \`docs/pages/\`.


# Examples (by language)

Each section below contains the canonical \`datamunge<sub>ex</sub>.\*\` example for that language.
During \`just prebuild-docs-pages\`, hidden shell blocks execute these examples
best-effort (printing \`SKIP\` when the relevant runtime/binding is unavailable).


## Sanity (executed)

    (+ 40 2)

    echo "Org Babel is executing blocks."


## C++

    #include <datamunge/datamunge.hpp>
    
    int main(void) {
      datamunge::hello();
      return 0;
    }

    Hello datamunge


## Python

    from pydatamunge import datamunge
    
    # dir(datamunge)
    
    datamunge.hello()
    
    dp = datamunge.DPair(1, 2)
    
    s = datamunge.SVector(3)
    s[0] = "1"
    s[1] = "2"
    s[2] = "3"
    # s[3] = "4"  # <- Gets index out of bounds error
    
    
    class TimesTwo(datamunge.Callback):
        def call(self, x):
            return x * 2.0
    
    
    class TimesThree(datamunge.Callback):
        def call(self, x):
            return x * 3.0
    
    
    # cb = TimesTwo()
    cb = TimesThree()
    print("call_with_callback(3.0) =", datamunge.call_with_callback(3.0, cb))
    v = datamunge.make_dvector(1.0, 2.0, 3.0)
    v2 = datamunge.map_dvector_with_callback(v, cb)
    print("sum_dvector(map_dvector_with_callback(1,2,3)) =", datamunge.sum_dvector(v2))

    Hello datamunge
    call_with_callback(3.0) = 9.0
    sum_dvector(map_dvector_with_callback(1,2,3)) = 18.0


## JavaScript (Node)

    // Local dev: build first (`just build-javascript`), then run this example.
    const datamunge = require("../../index.js");
    
    datamunge.hello();
    
    // Note: SWIG's Node/JavaScript backend does not support directors/virtual-method
    // overrides the same way as Python/Ruby/Perl. `Callback` can still be passed,
    // but the default implementation is identity.
    const cb = new datamunge.Callback();
    console.log("call_with_callback(3.0) =", datamunge.call_with_callback(3.0, cb));
    const v2 = datamunge.map_dvector_with_callback(datamunge.make_dvector(1.0, 2.0, 3.0), cb);
    console.log("sum_dvector(map_dvector_with_callback) =", datamunge.sum_dvector(v2));
    
    // Bridging: pass a JS function into native code (via C callback trampoline).
    console.log("call_with_function(3.0) =", datamunge.call_with_function(3.0, (x) => x * 2.0));
    const out = datamunge.map_array_with_function([1.0, 2.0, 3.0], (x) => x * 2.0);
    console.log("map_array_with_function([1,2,3]) =", out);

    Hello datamunge
    call_with_callback(3.0) = 3
    sum_dvector(map_dvector_with_callback) = 6
    call_with_function(3.0) = 6
    map_array_with_function([1,2,3]) = [ 2, 4, 6 ]


## R

    library(datamunger)
    
    datamunger::hello()
    
    # Can only coerce (can't use methods directly without a wrapper)
    v <- datamunger::DVector(c(0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 0.0))
    
    # This works as expected
    p <- datamunger::DPair(1.1, 2.2)
    p$first
    p$second
    
    # This works as expected
    p <- datamunger::IPair(1.1, 2.2) # converts to integer
    p$first
    p$second

    Hello datamunge
    NULL
    [1] 1.1
    [1] 2.2
    [1] 1
    [1] 2


## Ruby

    require "datamunge"
    
    Datamunge.hello
    
    pair = Datamunge.make_dpair(1.0, 2.0)  # std::pair<double, double> is a two-element Array
    puts "make_dpair: #{pair.inspect}, sum_dpair = #{Datamunge.sum_dpair(pair)}"
    
    # A Ruby subclass overrides the C++ Callback's virtual call().
    class TimesTwo < Datamunge::Callback
      def call(x) = x * 2.0
    end
    
    cb = TimesTwo.new
    puts "call_with_callback(3.0) = #{Datamunge.call_with_callback(3.0, cb)}"
    v = Datamunge.make_dvector(1.0, 2.0, 3.0)
    v2 = Datamunge.map_dvector_with_callback(v, cb)
    puts "sum_dvector(map_dvector_with_callback(1,2,3)) = #{Datamunge.sum_dvector(v2)}"

    Hello datamunge
    make_dpair: [1.0, 2.0], sum_dpair = 3.0
    call_with_callback(3.0) = 6.0
    sum_dvector(map_dvector_with_callback(1,2,3)) = 12.0


## Perl

    use strict;
    use warnings;
    
    use Datamunge;
    
    Datamunge::hello();
    
    my $dp = Datamunge::DPair->new(1.0, 2.0);
    print "DPair: First=" . $dp->swig_first_get() . ", Second=" . $dp->swig_second_get() . "\n";
    
    {
      package TimesTwo;
      our @ISA = ('Datamunge::Callback');
    
      sub call {
        my ($self, $x) = @_;
        return $x * 2.0;
      }
    }
    
    my $cb = TimesTwo->new();
    print "call_with_callback(3.0) = ", Datamunge::call_with_callback(3.0, $cb), "\n";
    my $v = Datamunge::make_dvector(1.0, 2.0, 3.0);
    my $v2 = Datamunge::map_dvector_with_callback($v, $cb);
    print "sum_dvector(map_dvector_with_callback(1,2,3)) = ", Datamunge::sum_dvector($v2), "\n";

    Hello datamunge
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

    Hello datamunge
    sum_dvector: 6
    sum_dpair: 4
    call_with_callback(3.0): 3
    sum_dvector(map_dvector_with_callback(1,2,3)): 6


## Lua

    local datamunge = require("datamunge")
    datamunge.hello()
    
    local v = datamunge.make_dvector(1.0, 2.0, 3.0)
    print("sum_dvector:", datamunge.sum_dvector(v))
    
    local p = datamunge.make_dpair(1.25, 2.75)
    print("sum_dpair:", datamunge.sum_dpair(p))
    
    local cb = datamunge.Callback()
    print("call_with_callback(3.0):", datamunge.call_with_callback(3.0, cb))
    local v2 = datamunge.map_dvector_with_callback(datamunge.make_dvector(1.0, 2.0, 3.0), cb)
    print("sum_dvector(map_dvector_with_callback(1,2,3)):", datamunge.sum_dvector(v2))

    Hello datamunge
    sum_dvector:	6.0
    sum_dpair:	4.0
    call_with_callback(3.0):	3.0
    sum_dvector(map_dvector_with_callback(1,2,3)):	6.0


## Tcl

    package require datamunge
    
    datamunge::hello
    
    set v [datamunge::make_dvector 1.0 2.0 3.0]
    puts "sum_dvector:\t[datamunge::sum_dvector $v]"
    
    set p [datamunge::make_dpair 1.25 2.75]
    puts "sum_dpair:\t[datamunge::sum_dpair $p]"
    
    set cb [datamunge::Callback new]
    puts "call_with_callback(3.0):\t[datamunge::call_with_callback 3.0 $cb]"
    set v2 [datamunge::map_dvector_with_callback [datamunge::make_dvector 1.0 2.0 3.0] $cb]
    puts "sum_dvector(map_dvector_with_callback(1,2,3)):\t[datamunge::sum_dvector $v2]"
    $cb destroy

    Hello datamunge
    sum_dvector:	6.0
    sum_dpair:	4.0
    call_with_callback(3.0):	3.0
    sum_dvector(map_dvector_with_callback(1,2,3)):	6.0


## Octave

    datamunge.hello ();
    
    v = datamunge.make_dvector (1.0, 2.0, 3.0);
    printf ('sum_dvector:\t%.1f\n', datamunge.sum_dvector (v));
    
    p = datamunge.make_dpair (1.25, 2.75);
    printf ('sum_dpair:\t%.1f\n', datamunge.sum_dpair (p));
    
    cb = datamunge.Callback ();
    printf ('call_with_callback(3.0):\t%.1f\n', datamunge.call_with_callback (3.0, cb));
    v2 = datamunge.map_dvector_with_callback (datamunge.make_dvector (1.0, 2.0, 3.0), cb);
    printf ('sum_dvector(map_dvector_with_callback(1,2,3)):\t%.1f\n', datamunge.sum_dvector (v2));

    Hello datamunge
    sum_dvector:	6.0
    sum_dpair:	4.0
    call_with_callback(3.0):	3.0
    sum_dvector(map_dvector_with_callback(1,2,3)):	6.0


## Guile

    (use-modules (oop goops) (datamunge))
    
    (hello)
    (format #t "sum-dvector = ~a\n" (sum-dvector (make-dvector 10.0 20.0 30.0)))  ; vectors are lists
    (format #t "sum-dpair = ~a\n" (sum-dpair (make-dpair 4.0 5.0)))               ; pairs are pairs
    (define-class <twice> (<callback>))                                             ; override a C++ virtual
    (define-method (call (self <twice>) x) (* 2 x))
    (format #t "call-with-callback = ~a\n" (call-with-callback 3.0 (make <twice>)))

    Hello datamunge
    sum-dvector = 60.0
    sum-dpair = 9.0
    call-with-callback = 6.0


## OCaml

    open Datamunge
    
    let () =
      hello ();
      let v = make_dvector 1.0 2.0 3.0 in
      Printf.printf "sum_dvector:\t%g\n" (sum_dvector v);
      let p = make_dpair 1.25 2.75 in
      Printf.printf "sum_dpair:\t%g\n" (sum_dpair p);
      (* An OCaml subclass overrides the C++ Callback's virtual call(). *)
      let cb = Callback.extend ~call:(fun _ x -> 2.0 *. x) () in
      Printf.printf "call_with_callback(3.0):\t%g\n" (call_with_callback 3.0 cb);
      let v2 = map_dvector_with_callback (make_dvector 1.0 2.0 3.0) cb in
      Printf.printf "sum_dvector(map_dvector_with_callback(1,2,3)):\t%g\n" (sum_dvector v2)

    Hello datamunge
    sum_dvector:	6
    sum_dpair:	4
    call_with_callback(3.0):	6
    sum_dvector(map_dvector_with_callback(1,2,3)):	12


## Go

    package main
    
    import (
            "datamunge"
            "fmt"
    )
    
    func main() {
            if err := datamunge.Hello(); err != nil {
                    panic(err)
            }
    
            v := datamunge.Must(datamunge.MakeDvector(1.0, 2.0, 3.0))
            fmt.Printf("sum_dvector:\t%v\n", datamunge.Must(datamunge.SumDvector(v)))
    
            p := datamunge.Must(datamunge.MakeDpair(1.25, 2.75))
            fmt.Printf("sum_dpair:\t%v\n", datamunge.Must(datamunge.SumDpair(p)))
    
            cb := datamunge.Must(datamunge.NewCallback())
            defer cb.Close()
            fmt.Printf("call_with_callback(3.0):\t%v\n", datamunge.Must(datamunge.CallWithCallback(3.0, cb)))
    }

    Hello datamunge
    sum_dvector:	6
    sum_dpair:	4
    call_with_callback(3.0):	3


# Using Datamunge

See \`README.md\` for language bindings and build/test commands.

