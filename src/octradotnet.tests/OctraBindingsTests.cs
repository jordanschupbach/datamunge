using Xunit;

public class OctraBindingsTests
{
    [Fact]
    public void Hello_IsCallable()
    {
        octra.hello();
    }

    [Fact]
    public void StdTemplates_AreUsable()
    {
        var p = new DPair(1.25, 2.5);
        Assert.Equal(1.25, p.first, 10);
        Assert.Equal(2.5, p.second, 10);

        var v = new DVector();
        v.Add(3.0);
        v.Add(4.5);
        Assert.Equal(2, v.Count);
        Assert.Equal(3.0, v[0], 10);
        Assert.Equal(4.5, v[1], 10);
    }
}
