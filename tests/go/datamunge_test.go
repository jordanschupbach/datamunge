// Smoke tests of the datamunge Go binding (just test-go): go test in the module tests/go, whose
// go.mod replaces datamunge with the committed src/datamungego/go/datamunge, linked against
// libdatamunge_go through CGO_LDFLAGS. The documentation checks read that committed module.
package tests

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"datamunge"
	"datamunge/data"
)

// doubler overrides the C++ Callback's virtual call() (a director).
type doubler struct{ *datamunge.Callback }

func (doubler) Call(x float64) float64 { return 2 * x }

// failing fails inside its override.
type failing struct{ *datamunge.Callback }

func (failing) Call(float64) (float64, error) { return 0, errors.New("from Go") }

// moduleDir is the committed generated module (the replace target in go.mod).
const moduleDir = "../../src/datamungego/go/datamunge"

func TestDatamunge(t *testing.T) {
	checks := 0
	check := func(ok bool, what string) {
		t.Helper()
		checks++
		if !ok {
			t.Fatalf("check failed: %s", what)
		}
	}
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}

	// Smoke test: call into the native library.
	must(datamunge.Hello())

	// std::vector<double> and std::pair<double, double> round trips.
	v := datamunge.Must(datamunge.MakeDvector(1.0, 2.0, 3.0))
	check(slices.Equal(v, []float64{1, 2, 3}), "std::vector<double> is a []float64")
	check(datamunge.Must(datamunge.SumDvector(v)) == 6, "SumDvector")
	check(datamunge.Must(datamunge.SumDvector([]float64{0.5, 1.5})) == 2, "slices convert to std::vector")
	p := datamunge.Must(datamunge.MakeDpair(1.25, 2.75))
	check(p.First == 1.25 && p.Second == 2.75, "std::pair is Pair")
	check(datamunge.Must(datamunge.SumDpair(p)) == 4, "SumDpair")

	// The C++ Callback class, and a Go type overriding its virtual call().
	cb := datamunge.Must(datamunge.NewCallback())
	defer cb.Close()
	check(datamunge.Must(datamunge.CallWithCallback(3, cb)) == 3, "native Callback")
	d := datamunge.Must(datamunge.NewCallbackFrom(&doubler{}))
	defer d.Close()
	check(datamunge.Must(datamunge.CallWithCallback(3, d)) == 6, "the Go override runs")
	mapped := datamunge.Must(datamunge.MapDvectorWithCallback([]float64{1, 2, 3}, d))
	check(datamunge.Must(datamunge.SumDvector(mapped)) == 12, "override from C++ loops")

	// A Go error inside an override reaches the caller as a CallbackError wrapping it.
	f := datamunge.Must(datamunge.NewCallbackFrom(&failing{}))
	defer f.Close()
	_, err := datamunge.CallWithCallback(1, f)
	var e *datamunge.Error
	check(errors.As(err, &e) && e.Kind == datamunge.CallbackError && e.Err != nil && e.Err.Error() == "from Go",
		"the Go error propagates")

	// C++ exceptions are *Error values with a kind.
	_, err = datamunge.Runif(1, 0, 3)
	check(errors.As(err, &e) && e.Kind == datamunge.InvalidArgument, "std::invalid_argument is InvalidArgument")

	// Classes and methods; Close releases the C++ object.
	frame := datamunge.Must(datamunge.NewDataFrame())
	must(frame.AddNumericColumn("x", []float64{1, 2, 3}))
	check(datamunge.Must(frame.Nrows()) == 3, "DataFrame")
	check(slices.Equal(datamunge.Must(frame.Columns()), []string{"x"}), "std::vector<std::string> is a []string")
	frame.Close()
	_, err = frame.Nrows()
	check(errors.As(err, &e) && e.Kind == datamunge.BindingError && frame.IsClosed(), "a closed object refuses calls")

	// The policy module datamunge::data is the subpackage datamunge/data.
	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	check(datamunge.Must(iris.Nrows()) == 150, "data.Iris")

	// The committed module carries the C++ documentation as godoc comments.
	source, err := os.ReadFile(filepath.Join(moduleDir, "datamunge.go"))
	must(err)
	seql := regexp.MustCompile(`((?://.*\n)+)func Seql\(`).FindSubmatch(source)
	check(seql != nil && strings.Contains(string(seql[1]), "evenly spaced"), "function godoc")
	sortBy := regexp.MustCompile(`((?://.*\n)+)func \(d \*DataFrame\) SortBy\(`).FindSubmatch(source)
	check(sortBy != nil && strings.Contains(string(sortBy[1]), "[DataFrame.SortByWithAscending]"), "method godoc")

	// go doc reads it (when go is on PATH, as under go test).
	if goTool, lookErr := exec.LookPath("go"); lookErr == nil {
		out, docErr := exec.Command(goTool, "doc", "datamunge.Seql").CombinedOutput()
		check(docErr == nil && strings.Contains(string(out), "evenly spaced"), "go doc datamunge.Seql: "+string(out))
	}

	t.Logf("all %d checks passed", checks)
}
