package classifiers;

import java.util.List;
import data.Instance;

public interface Model<F, L> {
    void train(List<Instance<F, L>> instances);
    List<L> test(List<Instance<F, L>> instances);
}
